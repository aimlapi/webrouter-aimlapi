"""AI/ML API Toolbox search and extract providers.

Talk to the Toolbox REST endpoint directly. Install with the ``search`` or
``extract`` extra. Read ``AIMLAPI_API_KEY`` unless a key is passed explicitly.

Both methods post to one endpoint, ``/v1/tools/run``, naming a tool and its
input: search runs ``web/search``, extract runs ``litescrape/web-fetch`` once
per URL. The reply carries the run's ``status`` and the tool's own ``output``.
The key goes only to ``tools.aimlapi.com``: a ``base_url`` naming another host
raises, and, as for every keyed HTTP provider, redirects are never followed.
"""

from __future__ import annotations

from typing import Any, ClassVar

import httpx

from web_router.constants import ENV_AIMLAPI_API_KEY, PROVIDER_AIMLAPI
from web_router.constants_aimlapi import (
    AIMLAPI_API_KEY_PREFIX,
    AIMLAPI_COMPLETED,
    AIMLAPI_EXTRACT_BODY_KEYS,
    AIMLAPI_EXTRACT_DEFAULT_FORMAT,
    AIMLAPI_EXTRACT_FORMATS,
    AIMLAPI_EXTRACT_TOOL,
    AIMLAPI_MAX_RESULTS,
    AIMLAPI_RUN_PATH,
    AIMLAPI_SEARCH_PARAMETERS,
    AIMLAPI_SEARCH_TOOL,
    AIMLAPI_TOOLS_BASE_URL,
    AIMLAPI_TOOLS_HOST,
    AIMLAPI_WAIT_MS,
    MSG_AIMLAPI_FOREIGN_HOST,
    MSG_AIMLAPI_NO_OUTPUT,
    MSG_AIMLAPI_PAGE_FAILED,
    MSG_AIMLAPI_RUN_FAILED,
)
from web_router.exceptions import ConfigurationError, ProviderError
from web_router.providers._http import HttpExtractProvider, HttpSearchProvider
from web_router.providers._utils import build_payload, compact
from web_router.types import (
    EXTRACT_FORMAT_ONLY,
    ExtractedDocument,
    ExtractFailure,
    ExtractFormat,
    ExtractRequest,
    ExtractResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)

_MAPPED_RESULT_FIELDS = frozenset({"url", "title", "snippet", "published_at"})
_MAPPED_DOCUMENT_FIELDS = frozenset({"url", "title", "content", "html", "text", "published_time"})


def _checked_base_url(base_url: str | None) -> str | None:
    """``base_url`` when it is unset or on :data:`AIMLAPI_TOOLS_HOST` over https; anything else raises."""
    if base_url is None:
        return None
    url = httpx.URL(base_url)
    if url.scheme != "https" or url.host != AIMLAPI_TOOLS_HOST:
        raise ConfigurationError(MSG_AIMLAPI_FOREIGN_HOST.format(home=AIMLAPI_TOOLS_HOST, host=url.host or base_url))
    return base_url


def _run_body(request: SearchRequest | ExtractRequest, tool: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """The ``/v1/tools/run`` body. ``provider_options`` reach its top level, as on every backend: ``max_cost_usd``,
    ``wait_ms``, or an ``input`` of their own, which replaces the mapped one whole."""
    return build_payload(request, tool=tool, input=compact(tool_input), wait_ms=AIMLAPI_WAIT_MS)


def _output(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The run's ``output``, or a ``ProviderError`` for a run that did not complete or failed in the tool."""
    status = payload.get("status")
    if status != AIMLAPI_COMPLETED or payload.get("is_error"):
        error = payload.get("error")
        message = error.get("message") if isinstance(error, dict) else error
        tool = payload.get("tool") or "?"
        raise ProviderError(name, MSG_AIMLAPI_RUN_FAILED.format(tool=tool, status=status, error=message or "no detail"))
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ProviderError(name, MSG_AIMLAPI_NO_OUTPUT.format(fields=", ".join(sorted(payload)) or "none"))
    return output


class AimlapiProvider(HttpSearchProvider):
    """Routes searches to the AI/ML API Toolbox ``web/search`` tool.

    A result's ``description`` and ``content`` are its ``snippet``; ``published_at``
    is its date when the tool reports one. The run's ``cost`` and ``attempts`` stay in ``raw``.
    """

    name: ClassVar[str] = PROVIDER_AIMLAPI
    supported_parameters: ClassVar[frozenset[str]] = AIMLAPI_SEARCH_PARAMETERS

    base_url: ClassVar[str] = AIMLAPI_TOOLS_BASE_URL
    search_path: ClassVar[str] = AIMLAPI_RUN_PATH
    env_api_key: ClassVar[str] = ENV_AIMLAPI_API_KEY
    api_key_prefix: ClassVar[str] = AIMLAPI_API_KEY_PREFIX

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=_checked_base_url(base_url), **kwargs)

    def _build_payload(self, request: SearchRequest) -> dict[str, Any]:
        count = min(request.max_results, AIMLAPI_MAX_RESULTS) if request.max_results else None
        tool_input = {"query": request.query, "max_results": count, "include_domains": request.include_domains}
        return _run_body(request, AIMLAPI_SEARCH_TOOL, tool_input)

    def _to_response(self, request: SearchRequest, payload: dict[str, Any]) -> SearchResponse:
        hits = _output(self.name, payload).get("results")
        if not isinstance(hits, list):
            raise ProviderError(self.name, MSG_AIMLAPI_NO_OUTPUT.format(fields="output without results"))
        return SearchResponse(
            provider=self.name,
            query=request.query,
            results=[self._to_result(hit) for hit in hits if isinstance(hit, dict) and hit.get("url")],
            raw=payload,
        )

    @staticmethod
    def _to_result(hit: dict[str, Any]) -> SearchResult:
        snippet = hit.get("snippet") or ""
        return SearchResult(
            url=hit["url"],
            title=hit.get("title") or "",
            description=snippet,
            content=snippet,
            published_date=hit.get("published_at") or None,
            extra={k: v for k, v in hit.items() if k not in _MAPPED_RESULT_FIELDS},
        )


class AimlapiExtractProvider(HttpExtractProvider):
    """Routes extracts to the AI/ML API Toolbox ``litescrape/web-fetch`` tool, one run per URL.

    The page body is read from the output key for the format asked for. A run
    that completes with no body -- the site answered 404, say -- is a failed URL
    carrying the tool's ``warning``, so ``auto`` can try another backend for it.
    """

    name: ClassVar[str] = PROVIDER_AIMLAPI
    supported_parameters: ClassVar[frozenset[str]] = EXTRACT_FORMAT_ONLY
    supported_formats: ClassVar[frozenset[ExtractFormat]] = frozenset(AIMLAPI_EXTRACT_FORMATS)
    default_format: ClassVar[ExtractFormat] = AIMLAPI_EXTRACT_DEFAULT_FORMAT

    base_url: ClassVar[str] = AIMLAPI_TOOLS_BASE_URL
    extract_path: ClassVar[str] = AIMLAPI_RUN_PATH
    env_api_key: ClassVar[str] = ENV_AIMLAPI_API_KEY
    api_key_prefix: ClassVar[str] = AIMLAPI_API_KEY_PREFIX

    def __init__(self, *, base_url: str | None = None, **kwargs: Any) -> None:
        super().__init__(base_url=_checked_base_url(base_url), **kwargs)

    def _build_payload(self, request: ExtractRequest, urls: list[str]) -> dict[str, Any]:
        respond_with = AIMLAPI_EXTRACT_FORMATS[self.format_of(request)]
        return _run_body(request, AIMLAPI_EXTRACT_TOOL, {"url": urls[0], "respond_with": respond_with})

    def _to_response(self, request: ExtractRequest, urls: list[str], payload: dict[str, Any]) -> ExtractResponse:
        body_format = self.format_of(request)
        output = _output(self.name, payload)
        url = output.get("url") or urls[0]
        content = output.get(AIMLAPI_EXTRACT_BODY_KEYS[body_format]) or ""
        if not content:
            warning = output.get("warning") or "empty page"
            failure = ExtractFailure(url=url, error=MSG_AIMLAPI_PAGE_FAILED.format(warning=warning))
            return ExtractResponse(provider=self.name, documents=[], failures=[failure], raw=[payload])
        document = ExtractedDocument(
            url=url,
            title=output.get("title") or "",
            content=content,
            format=body_format,
            published_date=output.get("published_time") or None,
            extra={k: v for k, v in output.items() if k not in _MAPPED_DOCUMENT_FIELDS},
        )
        return ExtractResponse(provider=self.name, documents=[document], raw=[payload])
