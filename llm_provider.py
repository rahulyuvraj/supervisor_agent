"""Unified LLM provider for the Supervisor Agent.

Primary: AWS Bedrock (Claude) when USE_BEDROCK=true.
Fallback: OpenAI GPT-4o.

All LLM calls across the supervisor agent flow through llm_complete().
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Load supervisor-agent-level .env (contains Bedrock + OpenAI keys)
_SUPERVISOR_ENV = Path(__file__).resolve().parent / ".env"
if _SUPERVISOR_ENV.is_file():
    load_dotenv(_SUPERVISOR_ENV, override=False)

logger = logging.getLogger(__name__)

_LLM_EXECUTOR = ThreadPoolExecutor(
    max_workers=int(os.getenv("LLM_MAX_CONCURRENCY", "10"))
)
_openai_lock = asyncio.Lock()
_openai_client = None


@dataclasses.dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    latency_ms: int
    fallback_used: bool
    intent: str = ""  # populated by callers for telemetry
    input_tokens: int = 0
    output_tokens: int = 0

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class LLMProviderError(Exception):
    """Raised when both Bedrock and OpenAI LLM providers fail."""


def _safe_json_parse(text: str) -> Any:
    """Parse JSON with resilience against Claude's backtick wrapping."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Strip markdown code fences
    stripped = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    stripped = re.sub(r'\n?\s*```\s*$', '', stripped, flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    # Extract between first { and last }
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(stripped[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("Failed to parse LLM response as JSON", text, 0)


@functools.lru_cache(maxsize=1)
def _get_bedrock_client():
    from botocore.config import Config as BotoConfig
    import boto3
    return boto3.client(
        "bedrock-runtime",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        config=BotoConfig(
            read_timeout=300,
            connect_timeout=60,
            retries={"max_attempts": 3},
        ),
    )


def _bedrock_invoke(messages: List[Dict], temperature: float,
                    max_tokens: int, system: Optional[str],
                    response_format: Optional[Dict]) -> str:
    """Synchronous Bedrock invoke_model — called via run_in_executor."""
    conversation = []
    sys_parts = []
    if system:
        sys_parts.append(system)
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            sys_parts.append(content)
        elif role in ("user", "assistant"):
            conversation.append({"role": role, "content": content})

    if response_format and response_format.get("type") == "json_object":
        sys_parts.append(
            "You must respond with valid JSON only. "
            "No markdown code fences, no explanation, no text outside the JSON."
        )

    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": conversation,
    }
    if sys_parts:
        body["system"] = "\n\n".join(sys_parts)

    client = _get_bedrock_client()
    model_id = os.getenv("BEDROCK_MODEL_ID",
                         "us.anthropic.claude-opus-4-5-20251101-v1:0")
    response = client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body),
    )
    resp_body = json.loads(response["body"].read())
    usage = resp_body.get("usage", {})
    text = resp_body.get("content", [{}])[0].get("text", "")
    return text, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


async def llm_complete(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    system: Optional[str] = None,
    response_format: Optional[Dict] = None,
) -> LLMResult:
    """Unified async LLM call — Bedrock primary, OpenAI fallback."""
    _ensure_config()
    use_bedrock = os.getenv("USE_BEDROCK", "").lower() == "true"
    bedrock_ok = use_bedrock and os.getenv("AWS_ACCESS_KEY_ID")

    if bedrock_ok:
        model_id = os.getenv("BEDROCK_MODEL_ID", "claude")
        t0 = time.monotonic()
        try:
            loop = asyncio.get_running_loop()
            text, in_tok, out_tok = await loop.run_in_executor(
                _LLM_EXECUTOR, _bedrock_invoke, messages, temperature,
                max_tokens, system, response_format,
            )
            ms = int((time.monotonic() - t0) * 1000)
            result = LLMResult(text=text, provider="bedrock", model=model_id,
                               latency_ms=ms, fallback_used=False,
                               input_tokens=in_tok, output_tokens=out_tok)
            logger.info("LLM [%s] %d in / %d out tokens, %dms",
                        result.provider, result.input_tokens,
                        result.output_tokens, result.latency_ms)
            return result
        except Exception as exc:
            logger.warning(f"Bedrock call failed ({exc}), falling back to OpenAI")

    # OpenAI fallback
    return await _openai_call(messages, temperature, max_tokens,
                              system, response_format)


async def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        async with _openai_lock:
            if _openai_client is None:
                from openai import AsyncOpenAI
                _openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _openai_client


async def _openai_call(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    system: Optional[str],
    response_format: Optional[Dict],
) -> LLMResult:
    client = await _get_openai_client()
    model = "gpt-4o"
    all_msgs = []
    if system:
        all_msgs.append({"role": "system", "content": system})
    all_msgs.extend(messages)

    kwargs: Dict[str, Any] = dict(model=model, temperature=temperature,
                                  max_tokens=max_tokens, messages=all_msgs)
    if response_format:
        kwargs["response_format"] = response_format

    t0 = time.monotonic()
    try:
        resp = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise LLMProviderError(f"OpenAI fallback failed: {exc}") from exc
    ms = int((time.monotonic() - t0) * 1000)
    text = resp.choices[0].message.content or ""
    in_tok = resp.usage.prompt_tokens if resp.usage else 0
    out_tok = resp.usage.completion_tokens if resp.usage else 0
    result = LLMResult(text=text, provider="openai", model=model,
                       latency_ms=ms, fallback_used=True,
                       input_tokens=in_tok, output_tokens=out_tok)
    logger.info("LLM [%s] %d in / %d out tokens, %dms",
                result.provider, result.input_tokens,
                result.output_tokens, result.latency_ms)
    return result


# ── Startup validation ──

def validate_llm_config():
    use_bedrock = os.getenv("USE_BEDROCK", "").lower() == "true"
    has_bedrock = bool(os.getenv("AWS_ACCESS_KEY_ID") and
                       os.getenv("AWS_SECRET_ACCESS_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    model_id = os.getenv("BEDROCK_MODEL_ID", "")

    if use_bedrock and has_bedrock:
        logger.info(f"LLM Provider: Bedrock ({model_id}) with OpenAI fallback")
    elif has_openai:
        if use_bedrock and not has_bedrock:
            logger.warning("USE_BEDROCK=true but AWS credentials missing — "
                           "using OpenAI only")
        else:
            logger.info("LLM Provider: OpenAI only (Bedrock not configured)")
    else:
        logger.warning("No LLM provider configured — both Bedrock and "
                       "OpenAI credentials missing")


_config_validated = False


def _ensure_config() -> None:
    global _config_validated
    if not _config_validated:
        validate_llm_config()
        _config_validated = True
