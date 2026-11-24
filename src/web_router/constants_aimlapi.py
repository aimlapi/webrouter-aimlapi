"""AI/ML API Toolbox search and extract: endpoint, tool ids and option mapping.

Kept apart from :mod:`web_router.constants`, as the other newer backends' are,
so that file stops growing. The env var and provider name stay there.
"""

from __future__ import annotations

from web_router.types import ExtractFormat

# POST /v1/tools/run (docs.aimlapi.com/quickstart/toolbox), with the key as a Bearer token.
# One endpoint runs every tool: the body names the tool and carries its input, and the
# reply holds the tool's own output under "output".

AIMLAPI_TOOLS_HOST = "tools.aimlapi.com"
"""The one host the key is sent to; a ``base_url`` naming another is refused."""

AIMLAPI_TOOLS_BASE_URL = f"https://{AIMLAPI_TOOLS_HOST}"
AIMLAPI_RUN_PATH = "/v1/tools/run"
AIMLAPI_API_KEY_PREFIX = "Bearer "

AIMLAPI_SEARCH_TOOL = "web/search"
"""The curated web search tool: titles, URLs, snippets and, where known, a date per result."""

AIMLAPI_EXTRACT_TOOL = "litescrape/web-fetch"
"""A page as Markdown, HTML or plain text, one URL per run."""

AIMLAPI_WAIT_MS = 55_000
"""How long a run may hold the request open before answering ``queued``: under the HTTP timeout."""

AIMLAPI_COMPLETED = "completed"
"""The one run ``status`` that carries an output; ``queued``, ``failed`` and the rest raise."""

AIMLAPI_SEARCH_PARAMETERS = frozenset({"max_results", "include_domains"})
"""What the search tool maps: ``max_results`` (at most :data:`AIMLAPI_MAX_RESULTS`) and ``include_domains``."""

AIMLAPI_MAX_RESULTS = 20
"""The most results one ``web/search`` run returns; its input schema refuses more."""

AIMLAPI_EXTRACT_FORMATS: dict[ExtractFormat, str] = {"markdown": "markdown", "html": "html", "text": "text"}
"""Normalized format -> ``respond_with``. The body comes back under a key named for it, ``content`` for markdown."""

AIMLAPI_EXTRACT_BODY_KEYS: dict[ExtractFormat, str] = {"markdown": "content", "html": "html", "text": "text"}
"""Normalized format -> the output key the page body is read from, checked against live runs."""

AIMLAPI_EXTRACT_DEFAULT_FORMAT: ExtractFormat = "markdown"

MSG_AIMLAPI_FOREIGN_HOST = "AI/ML API sends its key only to {home} over https; base_url names {host}"
MSG_AIMLAPI_RUN_FAILED = "AI/ML API tool run {tool} ended {status}: {error}"
MSG_AIMLAPI_NO_OUTPUT = "AI/ML API answered without an output object (fields: {fields})"
MSG_AIMLAPI_PAGE_FAILED = "AI/ML API could not read the page: {warning}"
