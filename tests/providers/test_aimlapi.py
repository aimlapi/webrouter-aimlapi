"""AI/ML API Toolbox provider tests.

``tests/fixtures/aimlapi/web_search.json`` is a reply recorded from the live
``POST /v1/tools/run`` (tool ``web/search``) on 2026-11-24, with the run id
replaced. The extract replies follow the shape of live ``litescrape/web-fetch``
runs on the same day: the page body under ``content`` for markdown, ``html`` or
``text`` for the others, and a ``warning`` with no body for a page that failed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest

from web_router.exceptions import ConfigurationError, ProviderError, UnsupportedParameterError
from web_router.providers.aimlapi import AimlapiExtractProvider, AimlapiProvider
from web_router.types import ExtractRequest, SearchRequest

RECORDED_SEARCH: dict[str, Any] = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "aimlapi" / "web_search.json").read_text(encoding="utf-8")
)
KEY = "test-key"


def _page(output: dict[str, Any]) -> dict[str, Any]:
    return {"tool": "litescrape/web-fetch", "status": "completed", "is_error": False, "output": output}


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> AimlapiProvider:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    return AimlapiProvider()


@pytest.fixture
def extractor(monkeypatch: pytest.MonkeyPatch) -> AimlapiExtractProvider:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    return AimlapiExtractProvider()


def test_search_runs_the_web_search_tool(provider: AimlapiProvider) -> None:
    body = provider._build_payload(SearchRequest(query="ai search", max_results=4, include_domains=["a.com"]))
    assert body == {
        "tool": "web/search",
        "input": {"query": "ai search", "max_results": 4, "include_domains": ["a.com"]},
        "wait_ms": 55_000,
    }
    assert provider._build_payload(SearchRequest(query="q", max_results=500))["input"]["max_results"] == 20
    assert provider._build_payload(SearchRequest(query="q"))["input"] == {"query": "q"}


def test_provider_options_reach_the_run_body(provider: AimlapiProvider) -> None:
    body = provider._build_payload(SearchRequest(query="q", provider_options={"max_cost_usd": 0.05}))
    assert body["max_cost_usd"] == 0.05


def test_the_key_travels_as_a_bearer_token_to_tools_aimlapi_com(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    sent: list[httpx.Request] = []
    provider = AimlapiProvider(
        transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200, json=RECORDED_SEARCH))
    )

    response = provider.search(SearchRequest(query="Model Context Protocol", max_results=3))

    assert sent[0].method == "POST"
    assert sent[0].url.scheme == "https"
    assert sent[0].url.host == "tools.aimlapi.com" and sent[0].url.path == "/v1/tools/run"
    assert sent[0].headers["authorization"] == f"Bearer {KEY}"
    assert len(response.results) == 3


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.example.com",
        "https://tools.aimlapi.com.evil.example",
        "https://evil.example/tools.aimlapi.com",
        "https://aimlapi.com",
        "https://api.aimlapi.com",
        "http://tools.aimlapi.com",
    ],
)
@pytest.mark.parametrize("cls", [AimlapiProvider, AimlapiExtractProvider])
def test_the_key_never_goes_to_another_host(base_url: str, cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    sent: list[httpx.Request] = []
    with pytest.raises(ConfigurationError, match=r"only to tools\.aimlapi\.com") as caught:
        cls(base_url=base_url, transport=httpx.MockTransport(lambda r: sent.append(r) or httpx.Response(200)))
    assert not sent
    assert KEY not in str(caught.value)


def test_the_exact_host_is_accepted_as_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    AimlapiProvider(base_url="https://tools.aimlapi.com")


def test_a_redirect_is_not_followed_so_the_key_stays_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.host == "tools.aimlapi.com":
            return httpx.Response(307, headers={"location": "https://evil.example/v1/tools/run"})
        return httpx.Response(200, json=RECORDED_SEARCH)

    provider = AimlapiProvider(transport=httpx.MockTransport(answer))
    with pytest.raises(ProviderError):
        provider.search(SearchRequest(query="q"))
    assert [request.url.host for request in sent] == ["tools.aimlapi.com"]
    with pytest.raises(ConfigurationError, match="redirect"):
        AimlapiProvider(follow_redirects=True)


@pytest.mark.parametrize(
    "option",
    [{"exclude_domains": ["b.com"]}, {"recency": "week"}, {"country": "US"}, {"locale": "en"}, {"effort": "low"}],
)
def test_an_unmapped_option_is_refused_rather_than_dropped(provider: AimlapiProvider, option: dict) -> None:
    with pytest.raises(UnsupportedParameterError):
        provider.search(SearchRequest(query="q", **option))


def test_the_recorded_reply_maps_from_output_results(provider: AimlapiProvider) -> None:
    response = provider._to_response(SearchRequest(query="Model Context Protocol"), RECORDED_SEARCH)

    first, second = response.results[0], response.results[1]
    assert first.url == "https://modelcontextprotocol.io/docs/2026-07-28/getting-started/intro"
    assert first.title == "What is the Model Context Protocol (MCP)?"
    assert first.description == first.content and first.content.startswith("MCP (Model Context Protocol)")
    assert first.published_date is None
    assert second.published_date == "2024-11-25T00:00:00.000Z"
    assert response.raw["cost"]["actual_usd"] == 0.0091


@pytest.mark.parametrize(
    "payload",
    [
        {"tool": "web/search", "status": "failed", "error": {"message": "upstream down"}},
        {"tool": "web/search", "status": "queued"},
        {"tool": "web/search", "status": "completed", "is_error": True, "error": "bad input"},
        {"tool": "web/search", "status": "completed"},
        {"tool": "web/search", "status": "completed", "output": {"items": []}},
    ],
)
def test_a_run_that_did_not_complete_is_an_error_not_an_empty_search(provider: AimlapiProvider, payload: dict) -> None:
    with pytest.raises(ProviderError):
        provider._to_response(SearchRequest(query="q"), payload)


def test_an_empty_results_list_is_a_search_that_found_nothing(provider: AimlapiProvider) -> None:
    payload = {"status": "completed", "output": {"results": []}}
    assert provider._to_response(SearchRequest(query="q"), payload).results == []


@pytest.mark.parametrize(
    ("asked", "sent"), [(None, "markdown"), ("markdown", "markdown"), ("html", "html"), ("text", "text")]
)
def test_extract_runs_web_fetch_with_the_format(extractor: AimlapiExtractProvider, asked: Any, sent: str) -> None:
    request = ExtractRequest(urls=["https://a.com"], format=asked) if asked else ExtractRequest(urls=["https://a.com"])
    body = extractor._build_payload(request, ["https://a.com"])
    assert body == {
        "tool": "litescrape/web-fetch",
        "input": {"url": "https://a.com", "respond_with": sent},
        "wait_ms": 55_000,
    }


@pytest.mark.parametrize(("fmt", "key"), [("markdown", "content"), ("html", "html"), ("text", "text")])
def test_the_page_body_is_read_from_the_key_for_its_format(
    extractor: AimlapiExtractProvider, fmt: Any, key: str
) -> None:
    payload = _page({"url": "https://a.com", "title": "A", key: "Body.", "published_time": "2026-01-01"})
    response = extractor._to_response(ExtractRequest(urls=["https://a.com"], format=fmt), ["https://a.com"], payload)
    (document,) = response.documents
    assert (document.url, document.title, document.content, document.format) == ("https://a.com", "A", "Body.", fmt)
    assert document.published_date == "2026-01-01"


def test_a_page_with_no_body_is_a_failed_url_carrying_the_warning(extractor: AimlapiExtractProvider) -> None:
    payload = _page({"url": "https://a.com/x", "content": "", "warning": "Target URL returned error 404"})
    response = extractor._to_response(ExtractRequest(urls=["https://a.com/x"]), ["https://a.com/x"], payload)
    assert response.documents == []
    assert [failure.url for failure in response.failures] == ["https://a.com/x"]
    assert "404" in response.failures[0].error


def test_each_url_is_its_own_run_and_a_failed_one_does_not_sink_the_rest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIMLAPI_API_KEY", KEY)
    hosts: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        assert request.headers["authorization"] == f"Bearer {KEY}"
        url = json.loads(request.content)["input"]["url"]
        output = {"url": url, "content": "Body."} if url == "https://a.com" else {"url": url, "warning": "blocked"}
        return httpx.Response(200, json=_page(output))

    extractor = AimlapiExtractProvider(transport=httpx.MockTransport(answer))
    response = extractor.extract(ExtractRequest(urls=["https://a.com", "https://b.com"]))

    assert hosts == ["tools.aimlapi.com", "tools.aimlapi.com"]
    assert [document.url for document in response.documents] == ["https://a.com"]
    assert [failure.url for failure in response.failures] == ["https://b.com"]


@pytest.mark.integration
def test_live_search_returns_web_results() -> None:
    if not os.environ.get("AIMLAPI_API_KEY"):
        pytest.skip("AIMLAPI_API_KEY not set")
    response = AimlapiProvider().search(SearchRequest(query="python asyncio tutorial", max_results=3))
    assert 0 < len(response.results) <= 3
    assert all(result.url.startswith("http") for result in response.results)


@pytest.mark.integration
def test_live_extract_returns_markdown() -> None:
    if not os.environ.get("AIMLAPI_API_KEY"):
        pytest.skip("AIMLAPI_API_KEY not set")
    response = AimlapiExtractProvider().extract(ExtractRequest(urls=["https://example.com"]))
    assert response.documents and "Example Domain" in response.documents[0].content
