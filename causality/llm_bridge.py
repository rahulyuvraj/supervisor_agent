"""
llm_bridge.py — Bridge between the supervisor LLM and the causality pipeline.

Provides ``CausalityLLMBridge`` which implements the ``complete()`` protocol
expected by all intelligence.py components (matching the demo's ClaudeClient
interface).  When constructed with a ``call_llm_fn`` it delegates to the
supervisor's Bedrock endpoint; otherwise it falls back to the deterministic
``LocalClient`` from intelligence.py.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

log = logging.getLogger("causal_platform")


class CausalityLLMBridge:
    """Duck-typed LLM client compatible with intelligence.py components.

    Parameters
    ----------
    call_llm_fn : callable, optional
        ``call_llm_fn(user_msg, system_prompt, max_tokens) -> str``
        Typically the supervisor's ``call_llm_api`` wrapper around Bedrock.
        If *None*, all calls are routed to the deterministic ``LocalClient``.
    """

    def __init__(self, call_llm_fn: Optional[Callable[..., str]] = None):
        self._call_llm_fn = call_llm_fn
        self._local = None  # lazy

    # ── public interface ──────────────────────────────────────────────────

    def complete(
        self,
        user_msg: str,
        system_prompt: str,
        max_tokens: int,
        as_json: bool = True,
    ) -> Any:
        """Call the LLM and return parsed JSON (default) or raw text.

        Falls back to ``LocalClient`` when *call_llm_fn* is absent or when
        the LLM call raises.
        """
        if self._call_llm_fn is not None:
            try:
                raw = self._call_llm_fn(user_msg, system_prompt, max_tokens)
                if as_json:
                    return self._parse_json(raw)
                return raw
            except Exception as exc:
                log.warning("LLM bridge call failed: %s — using local fallback", exc)
        return self._local_complete(user_msg, system_prompt, max_tokens, as_json)

    # ── internals ─────────────────────────────────────────────────────────

    def _local_complete(self, user_msg, system_prompt, max_tokens, as_json):
        if self._local is None:
            from .intelligence import LocalClient
            self._local = LocalClient()
        return self._local.complete(user_msg, system_prompt, max_tokens, as_json=as_json)

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """Strip markdown code fences and parse JSON."""
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return {"raw": text}
