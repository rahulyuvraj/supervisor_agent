"""Base class for all async API adapters.

Provides httpx client management, retry with exponential back-off,
rate-limiting, response caching, and auto-registration into
API_ADAPTER_REGISTRY via __init_subclass__.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, ClassVar, Dict, Optional, Type

import httpx

from .cache import TTLCache
from .config import APIConfig
from .rate_limiter import AsyncRateLimiter

logger = logging.getLogger(__name__)

# Auto-populated by subclasses declaring `service_name`
API_ADAPTER_REGISTRY: Dict[str, Type["BaseAPIAdapter"]] = {}


class TransientError(Exception):
    """Retriable HTTP error (429, 5xx)."""


class PermanentError(Exception):
    """Non-retriable HTTP error (4xx except 429)."""


class AdapterDisabledError(Exception):
    """Adapter gated off by a feature flag."""


class BaseAPIAdapter:
    """Async adapter base with built-in HTTP resilience.

    Subclasses set ``service_name`` (auto-registers) and ``base_url``,
    then call ``_request()`` for all HTTP work.
    """

    service_name: ClassVar[str] = ""
    base_url: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.service_name:
            API_ADAPTER_REGISTRY[cls.service_name] = cls

    def __init__(self, config: APIConfig, cache: Optional[TTLCache] = None) -> None:
        self.config = config
        self._cache = cache or TTLCache(
            max_size=config.cache_max_size,
            default_ttl=config.default_cache_ttl,
        )
        rps = self._resolve_rps()
        self._limiter = AsyncRateLimiter(requests_per_second=rps)
        self._client: Optional[httpx.AsyncClient] = None

    # ── Lifecycle ──

    async def __aenter__(self) -> "BaseAPIAdapter":
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.config.default_timeout,
            follow_redirects=True,
            headers={"Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    # ── Core request method ──

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Any] = None,
        data: Optional[Any] = None,
        content: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        cache_ttl: Optional[int] = None,
        skip_cache: bool = False,
    ) -> Any:
        """Make an HTTP request with retry, rate-limit and caching.

        Returns parsed JSON for JSON responses, or raw text otherwise.
        """
        if self._client is None:
            raise RuntimeError(
                f"{self.service_name}: adapter used outside async context manager"
            )

        # Cache lookup (GET only, no body)
        cache_key = None
        if method.upper() == "GET" and not skip_cache:
            cache_key = TTLCache.make_key(method, path, params)
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            async with self._limiter:
                try:
                    resp = await self._client.request(
                        method,
                        path,
                        params=params,
                        json=json,
                        data=data,
                        content=content,
                        headers=headers,
                    )
                except httpx.TransportError as exc:
                    last_exc = TransientError(str(exc))
                    await self._backoff(attempt)
                    continue

                if resp.status_code == 429 or resp.status_code >= 500:
                    last_exc = TransientError(
                        f"HTTP {resp.status_code} from {self.service_name}"
                    )
                    await self._backoff(attempt)
                    continue

                if 400 <= resp.status_code < 500:
                    raise PermanentError(
                        f"HTTP {resp.status_code} from {self.service_name}: "
                        f"{resp.text[:300]}"
                    )

                result = self._parse_response(resp)

                if cache_key is not None:
                    self._cache.set(cache_key, result, ttl=cache_ttl)

                return result

        raise last_exc or TransientError(f"{self.service_name}: retries exhausted")

    # ── Helpers ──

    @staticmethod
    def _parse_response(resp: httpx.Response) -> Any:
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        return resp.text

    @staticmethod
    async def _backoff(attempt: int) -> None:
        await asyncio.sleep(min(2 ** (attempt - 1), 16))

    def _resolve_rps(self) -> float:
        """Map service_name → config rate-limit field."""
        mapping = {
            "reactome": self.config.reactome_rps,
            "kegg": self.config.kegg_rps,
            "chembl": self.config.chembl_rps,
            "openfda": self.config.openfda_rps,
            "dgidb": self.config.dgidb_rps,
            "string": self.config.string_rps,
            "pubchem": self.config.pubchem_rps,
            "ensembl": self.config.ensembl_rps,
            "clinical_trials": self.config.clinical_trials_rps,
        }
        return mapping.get(self.service_name, 5.0)
