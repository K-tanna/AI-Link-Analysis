"""Offline tests: the network and the LLM are mocked, so no API keys are needed."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.errors import (
    BlockedURLError,
    InvalidURLError,
    LLMError,
    LLMOutputError,
    SocialContentUnavailableError,
    WebsiteFetchError,
)
from app.main import app, get_analysis_service
from app.services import llm_service
from app.services.analysis_service import AnalysisService
from app.services.llm_service import LLMResult, LLMService, extract_json
from app.services.website_extractor import WebsiteExtractor, parse_html, pick_priority_links
from app.utils import validators
from app.utils.social import detect_social, SocialInfo
from app.utils.validators import normalize_url, validate_public_url

# --------------------------------------------------------------------------- fixtures / fakes
HOME_HTML = """<html lang="en"><head><title>Acme Web Studio | Websites for Small Businesses</title>
<meta name="description" content="Acme builds affordable websites and online stores for small businesses in Ahmedabad.">
<script type="application/ld+json">{"@type":"Organization","name":"Acme Web Studio"}</script></head>
<body><nav><a href="/about">About</a><a href="/services">Services</a><a href="/contact">Contact</a></nav>
<h1>Affordable websites for small businesses</h1>
<h2>Web Design</h2><p>We design fast, mobile-friendly websites.</p>
<h2>E-commerce</h2><p>Online stores with payment integration.</p>
<img src="a.png"><img src="b.png" alt="team">
<script>var secret = "SHOULD_NOT_APPEAR";</script>
<footer>Based in Ahmedabad, India</footer></body></html>"""

ABOUT_HTML = "<html><head><title>About Acme</title></head><body><h1>About us</h1><p>Founded in 2015, Acme helps local shops go online.</p></body></html>"
SERVICES_HTML = "<html><head><title>Services</title></head><body><h1>Our services</h1><p>Web design, e-commerce, SEO and maintenance plans.</p></body></html>"
RIVAL_HTML = "<html><head><title>Rival {n} - Enterprise Web Agency</title><meta name='description' content='Enterprise web development.'></head><body><h1>Rival</h1></body></html>"


def site_handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "example.com":
        pages = {"/": HOME_HTML, "/about": ABOUT_HTML, "/services": SERVICES_HTML}
        if path in pages:
            return httpx.Response(200, text=pages[path], headers={"content-type": "text/html; charset=utf-8"})
        return httpx.Response(404)
    if host == "redirector.com":
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
    if host == "blocked.com":
        return httpx.Response(403)
    if host == "files.com":
        return httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"})
    if host in ("rival-one.com", "rival-two.com"):
        return httpx.Response(200, text=RIVAL_HTML.format(n=host), headers={"content-type": "text/html"})
    if host in ("instagram.com", "www.instagram.com"):
        if path == "/crestspot":
            html = (
                "<html><head>"
                "<title>Crest Spot (@crestspot) • Instagram photos and videos</title>"
                '<meta property="og:title" content="Crest Spot (@crestspot) • Instagram photos and videos">'
                '<meta property="og:description" content="1,200 Followers, 350 Following, 45 Posts - Crest Spot is a creative agency based in Mumbai crafting web and brand experiences.">'
                "</head><body><p>Crest Spot on Instagram</p></body></html>"
            )
            return httpx.Response(200, text=html, headers={"content-type": "text/html; charset=utf-8"})
        if path == "/loginwall":
            return httpx.Response(302, headers={"location": "https://www.instagram.com/accounts/login/"})
        if path == "/accounts/login/":
            return httpx.Response(200, text="<html><head><title>Login • Instagram</title></head><body>Log in to see photos</body></html>", headers={"content-type": "text/html; charset=utf-8"})
        if path == "/emptyprofile":
            return httpx.Response(200, text="<html><head><title>Instagram</title></head><body></body></html>", headers={"content-type": "text/html; charset=utf-8"})
    return httpx.Response(404)


@pytest.fixture(autouse=True)
def fake_dns(monkeypatch):
    """Resolve every hostname to a public IP, except 'internal.*' which resolves to a private one."""

    async def resolve(host, port):
        return ["10.0.0.5"] if host.startswith("internal") else ["93.184.216.34"]

    monkeypatch.setattr(validators, "resolve_host", resolve)


def make_settings(**kw) -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        llm_input_price_per_m=0.10,
        llm_output_price_per_m=0.50,
        **kw,
    )


PROFILE_OK = {
    "brand": {"name": "Acme Web Studio", "category": "Web Development", "summary": "Builds websites for small businesses."},
    "brand_overview": "Acme builds affordable websites and online stores.",
    "main_products_services": ["Web design", "E-commerce"],
    "competitor_candidates": [
        {"name": "Acme Web Studio", "website": "https://example.com", "reason": "self - must be dropped"},
        {"name": "Rival One", "website": "https://rival-one.com", "reason": "same market"},
        {"name": "Rival Two", "website": "https://rival-two.com", "reason": "same market"},
        {"name": "Rival Three", "website": "https://rival-three.com", "reason": "same market"},
    ],
    "target_search_queries": [
        {"query": "affordable website design Ahmedabad", "platforms": ["Google"], "intent": "Service Search"},
        {"query": "web design agency for small business", "platforms": ["Facebook", "Twitter"], "intent": "Discovery"},
        {"query": "small business website ideas", "platforms": ["Instagram"], "intent": "Inspiration"},
        {"query": "web development company India", "platforms": ["LinkedIn"], "intent": "B2B"},
        {"query": "need a website for my shop", "platforms": ["x"], "intent": "Service Discovery"},
    ],
    "seo_improvements": {
        "primary_keyword": "web design Ahmedabad",
        "secondary_keywords": ["small business websites"],
        "recommended_title": "Affordable Web Design in Ahmedabad | Acme",
        "meta_description": "Acme builds fast, affordable websites and online stores for small businesses.",
        "supporting_content": [{"type": "FAQ", "title": "How much does a website cost?", "rationale": "high intent"}],
    },
    "top_services": [
        {"name": f"Service {i}", "description": "d", "source": "website" if i < 3 else "inferred"} for i in range(5)
    ],
    "target_audience": {"primary_audience": ["Small business owners"], "industries": ["Retail"]},
}

COMPETITORS_OK = {
    "brand_name": "Acme Web Studio",
    "competitors": [
        {"name": "Rival One", "website": "https://rival-one.com", "strengths": ["Brand"], "weaknesses": ["Price"]},
        {"name": "Rival Two", "website": "https://rival-two.com", "strengths": ["Scale"], "weaknesses": ["Support"]},
        {"name": "Rival Three", "website": "https://rival-three.com", "strengths": ["Reach"], "weaknesses": ["Speed"]},
    ],
    "comparison": [{"aspect": "Target Market", "your_brand": "SMBs", "competitors": {"Rival One": "Enterprise"}}],
    "strengths": ["Affordable"],
    "opportunities": ["Improve visibility"],
}


class FakeLLM:
    """Scripted LLM. Each list is consumed in order; the last item repeats. Exceptions are raised."""

    provider_name, model_name = "fake", "fake-model"

    def __init__(self, profile=None, competitors=None):
        self.scripts = {"profile": list(profile or [PROFILE_OK]), "competitors": list(competitors or [COMPETITORS_OK])}
        self.calls = {"profile": 0, "competitors": 0}
        self.configured = True

    def ensure_configured(self):
        if not self.configured:
            from app.errors import LLMNotConfiguredError

            raise LLMNotConfiguredError("No API key configured.")

    async def complete_json(self, system, user, max_tokens=None):
        stage = "profile" if "[stage: profile]" in user else "competitors"
        self.calls[stage] += 1
        script = self.scripts[stage]
        item = script.pop(0) if len(script) > 1 else script[0]
        if isinstance(item, Exception):
            raise item
        return LLMResult(item, 1000, 500, 10, "fake", "fake-model")


def build_service(llm=None) -> AnalysisService:
    settings = make_settings()
    extractor = WebsiteExtractor(settings, transport=httpx.MockTransport(site_handler))
    return AnalysisService(settings, extractor=extractor, llm=llm or FakeLLM())


@pytest.fixture
def client_factory():
    def make(service):
        app.dependency_overrides[get_analysis_service] = lambda: service
        return TestClient(app, raise_server_exceptions=False)

    yield make
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------- URL validation / SSRF
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://127.0.0.1:80/admin",
        "https://10.1.2.3/",
        "http://192.168.0.1/",
        "http://172.16.5.5/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://0.0.0.0/",
        "http://100.64.0.1/",  # CGNAT
        "http://internal.corp.example/",  # hostname that resolves to a private IP
    ],
)
async def test_private_addresses_are_blocked(url):
    with pytest.raises(BlockedURLError):
        await validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    ["", "   ", "ftp://example.com", "file:///etc/passwd", "javascript:alert(1)", "http://user:pw@example.com/",
     "https://exa mple.com", "localhost", "http://", "https://" + "a" * 3000 + ".com"],
)
async def test_invalid_urls_are_rejected(url):
    with pytest.raises((InvalidURLError, BlockedURLError)):
        await validate_public_url(url)


async def test_non_standard_port_is_blocked():
    with pytest.raises(BlockedURLError):
        await validate_public_url("https://example.com:6379/")


def test_normalize_adds_scheme_and_strips_fragment():
    assert normalize_url("Example.com/path#frag") == "https://example.com/path"
    assert normalize_url("https://example.com") == "https://example.com/"


async def test_public_url_passes():
    assert await validate_public_url("https://example.com") == "https://example.com/"


# --------------------------------------------------------------------------- extraction
def test_parse_html_extracts_structure_and_strips_noise():
    page = parse_html("https://example.com/", HOME_HTML)
    assert page.title.startswith("Acme Web Studio")
    assert "affordable websites" in page.meta_description
    assert "Services" in page.nav_links
    assert "H1: Affordable websites for small businesses" in page.headings
    assert "Web Design" in page.text
    assert "SHOULD_NOT_APPEAR" not in page.text  # scripts stripped
    assert page.jsonld[0]["name"] == "Acme Web Studio"
    assert "Ahmedabad" in page.footer_text
    assert page.seo_facts["h1_count"] == 1
    assert page.seo_facts["images_total"] == 2
    assert page.seo_facts["images_missing_alt"] == 1
    assert page.seo_facts["has_structured_data"] is True


def test_priority_links_prefer_about_and_services():
    page = parse_html("https://example.com/", HOME_HTML)
    links = pick_priority_links(page, 2)
    assert set(links) == {"https://example.com/about", "https://example.com/services"}


async def test_extract_fetches_extra_pages():
    extractor = WebsiteExtractor(make_settings(), transport=httpx.MockTransport(site_handler))
    content = await extractor.extract("https://example.com")
    assert len(content.pages) == 3
    assert any("Founded in 2015" in p["text"] for p in content.pages)


async def test_redirect_to_private_ip_is_blocked():
    extractor = WebsiteExtractor(make_settings(), transport=httpx.MockTransport(site_handler))
    with pytest.raises(BlockedURLError):
        await extractor.extract("https://redirector.com")


async def test_bot_blocking_site_gives_clear_error():
    extractor = WebsiteExtractor(make_settings(), transport=httpx.MockTransport(site_handler))
    with pytest.raises(WebsiteFetchError, match="refused automated access"):
        await extractor.extract("https://blocked.com")


async def test_non_html_content_is_rejected():
    extractor = WebsiteExtractor(make_settings(), transport=httpx.MockTransport(site_handler))
    with pytest.raises(WebsiteFetchError, match="HTML"):
        await extractor.extract("https://files.com")


# --------------------------------------------------------------------------- LLM client
def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} Thanks') == {"a": {"b": 2}}
    with pytest.raises(LLMOutputError):
        extract_json("no json at all")
    with pytest.raises(LLMOutputError):
        extract_json('{"a": 1')  # truncated


def llm_service_with(handler, provider, model="m", **kw) -> LLMService:
    settings = Settings(_env_file=None, llm_provider=provider, llm_model=model, llm_max_retries=1,
                        **{f"{provider}_api_key": "k"}, **kw)
    return LLMService(settings, transport=httpx.MockTransport(handler))


async def test_openai_provider_parses_content_and_usage():
    def handler(req):
        body = json.loads(req.content)
        assert body["response_format"] == {"type": "json_object"}
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}],
                                         "usage": {"prompt_tokens": 11, "completion_tokens": 7}})

    res = await llm_service_with(handler, "openai").complete_json("s", "u")
    assert res.data == {"ok": True} and (res.input_tokens, res.output_tokens) == (11, 7)


async def test_anthropic_provider_parses_content_and_usage():
    def handler(req):
        assert req.headers["x-api-key"] == "k" and "anthropic-version" in req.headers
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"ok": 1}'}],
                                         "usage": {"input_tokens": 5, "output_tokens": 3}})

    res = await llm_service_with(handler, "anthropic").complete_json("s", "u")
    assert res.data == {"ok": 1} and (res.input_tokens, res.output_tokens) == (5, 3)


async def test_gemini_provider_counts_thinking_tokens_as_output():
    def handler(req):
        assert req.headers["x-goog-api-key"] == "k"
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "hidden", "thought": True}, {"text": '{"ok": 2}'}]}}],
            "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 4, "thoughtsTokenCount": 6}})

    res = await llm_service_with(handler, "gemini").complete_json("s", "u")
    assert res.data == {"ok": 2} and (res.input_tokens, res.output_tokens) == (9, 10)


async def test_transient_errors_are_retried(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}})

    await llm_service_with(handler, "openai").complete_json("s", "u")
    assert attempts["n"] == 2


async def test_auth_failure_is_not_retried_and_does_not_leak_body():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(401, text="Incorrect API key provided: sk-secret")

    with pytest.raises(LLMError) as exc:
        await llm_service_with(handler, "openai").complete_json("s", "u")
    assert calls["n"] == 1 and "sk-secret" not in exc.value.message


async def test_fallback_provider_used_when_primary_fails(monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm_service.asyncio, "sleep", no_sleep)

    def handler(req):
        if "openai.com" in str(req.url):
            return httpx.Response(503)
        return httpx.Response(200, json={"content": [{"type": "text", "text": '{"via": "fallback"}'}], "usage": {}})

    settings = Settings(_env_file=None, llm_provider="openai", llm_model="m", openai_api_key="a",
                        fallback_provider="anthropic", fallback_model="n", anthropic_api_key="b", llm_max_retries=0)
    res = await LLMService(settings, transport=httpx.MockTransport(handler)).complete_json("s", "u")
    assert res.data == {"via": "fallback"} and res.provider == "anthropic"


# --------------------------------------------------------------------------- API endpoint
def test_successful_analysis(client_factory):
    llm = FakeLLM()
    client = client_factory(build_service(llm))
    r = client.post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["success"] is True and body["website"] == "https://example.com/"
    assert body["brand"]["name"] == "Acme Web Studio"

    a = body["analysis"]
    assert set(a) == {"competitor_analysis", "target_search_queries", "seo_improvements", "top_services", "target_audience"}
    assert len(a["top_services"]) == 5
    assert len(a["target_search_queries"]) == 5
    # platform aliases normalised, and no invented volumes
    platforms = {p for q in a["target_search_queries"] for p in q["platforms"]}
    assert platforms == {"Google", "Facebook", "Instagram", "LinkedIn", "X"}
    assert all(q["search_volume"] is None for q in a["target_search_queries"])

    # competitor provenance is set by code: checked homepages are labelled, unreachable ones are not
    comps = {c["name"]: c["source"] for c in a["competitor_analysis"]["competitors"]}
    assert comps["Rival One"] == "ai_suggestion_homepage_checked"
    assert comps["Rival Three"] == "ai_suggestion"

    m = body["meta"]
    assert (m["input_tokens"], m["output_tokens"]) == (2000, 1000)
    assert m["estimated_cost_usd"] == pytest.approx(0.0007)
    assert len(m["pages_analyzed"]) == 3 and m["disclaimers"] and m["cached"] is False
    assert llm.calls == {"profile": 1, "competitors": 1}


def test_self_is_not_listed_as_competitor(client_factory):
    seen = {}

    class Spy(FakeLLM):
        async def complete_json(self, system, user, max_tokens=None):
            if "[stage: competitors]" in user:
                seen["prompt"] = user
            return await super().complete_json(system, user, max_tokens)

    client = client_factory(build_service(Spy()))
    assert client.post("/link-analysis", json={"url": "https://example.com"}).status_code == 200
    competitors_block = seen["prompt"].split("<competitors>")[1]
    assert "example.com" not in competitors_block and "rival-one.com" in competitors_block


def test_second_request_is_served_from_cache(client_factory):
    llm = FakeLLM()
    client = client_factory(build_service(llm))
    client.post("/link-analysis", json={"url": "https://example.com"})
    r = client.post("/link-analysis", json={"url": "https://example.com/"})
    assert r.json()["meta"]["cached"] is True
    assert llm.calls == {"profile": 1, "competitors": 1}


@pytest.mark.parametrize(
    "payload,status,code",
    [
        ({"url": "http://127.0.0.1:4000/"}, 400, "blocked_url"),
        ({"url": "http://169.254.169.254/"}, 400, "blocked_url"),
        ({"url": "https://redirector.com"}, 400, "blocked_url"),
        ({"url": "not a url"}, 400, "invalid_url"),
        ({"url": "ftp://example.com"}, 400, "invalid_url"),
        ({"url": "https://blocked.com"}, 422, "website_unreachable"),
        ({"url": "https://files.com"}, 422, "website_unreachable"),
        ({}, 422, "invalid_request"),
        ({"url": 123}, 422, "invalid_request"),
    ],
)
def test_error_responses_are_structured(client_factory, payload, status, code):
    client = client_factory(build_service())
    r = client.post("/link-analysis", json=payload)
    assert r.status_code == status, r.text
    body = r.json()
    assert body["success"] is False and body["error"]["code"] == code and body["error"]["message"]


def test_llm_not_configured_returns_503(client_factory):
    llm = FakeLLM()
    llm.configured = False
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "llm_not_configured"


def test_invalid_llm_output_is_retried_once_then_succeeds(client_factory):
    broken = {k: v for k, v in PROFILE_OK.items() if k != "top_services"}
    llm = FakeLLM(profile=[broken, PROFILE_OK])
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 200 and llm.calls["profile"] == 2


def test_persistently_invalid_llm_output_returns_502(client_factory):
    broken = {k: v for k, v in PROFILE_OK.items() if k != "top_services"}
    llm = FakeLLM(profile=[broken])
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "llm_invalid_output"
    assert llm.calls["profile"] == 2  # exactly one retry, not an endless loop


def test_malformed_json_from_llm_is_retried(client_factory):
    llm = FakeLLM(profile=[LLMOutputError("malformed"), PROFILE_OK])
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 200 and llm.calls["profile"] == 2


def test_provider_outage_returns_502(client_factory):
    llm = FakeLLM(profile=[LLMError("openai unavailable after retries")])
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://example.com"})
    assert r.status_code == 502 and r.json()["error"]["code"] == "llm_unavailable"


def test_health_endpoint(client_factory):
    assert client_factory(build_service()).get("/health").json() == {"status": "ok"}


# --------------------------------------------------------------------------- Social Profile Tests
def test_detect_social_platforms():
    # Instagram
    s1 = detect_social("https://www.instagram.com/crestspot")
    assert s1 is not None and s1.platform == "Instagram" and s1.handle == "crestspot"

    # Instagram reserved segment (e.g. reels)
    s2 = detect_social("https://www.instagram.com/reel/C3abc123")
    assert s2 is not None and s2.platform == "Instagram" and s2.handle is None

    # X / Twitter
    s3 = detect_social("https://x.com/openai")
    assert s3 is not None and s3.platform == "X" and s3.handle == "openai"

    s4 = detect_social("https://twitter.com/elonmusk")
    assert s4 is not None and s4.platform == "X" and s4.handle == "elonmusk"

    # YouTube
    s5 = detect_social("https://www.youtube.com/@mkbhd")
    assert s5 is not None and s5.platform == "YouTube" and s5.handle == "mkbhd"

    s6 = detect_social("https://www.youtube.com/c/Veritasium")
    assert s6 is not None and s6.platform == "YouTube" and s6.handle == "Veritasium"

    # LinkedIn
    s7 = detect_social("https://www.linkedin.com/company/google")
    assert s7 is not None and s7.platform == "LinkedIn" and s7.handle == "google"

    # Facebook
    s8 = detect_social("https://www.facebook.com/meta")
    assert s8 is not None and s8.platform == "Facebook" and s8.handle == "meta"

    # Standard website (not social)
    assert detect_social("https://example.com/crestspot") is None
    assert detect_social("https://crestspot.com") is None


def test_social_profile_analysis_success(client_factory):
    llm = FakeLLM()
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://www.instagram.com/crestspot"})
    assert r.status_code == 200
    data = r.json()
    meta = data["meta"]
    assert meta["source_type"] == "social_profile"
    assert meta["platform"] == "Instagram"
    assert meta["data_quality"] == "limited"
    assert len(meta["warnings"]) > 0
    assert "limited public metadata" in meta["warnings"][0]


def test_social_profile_login_wall_returns_422(client_factory):
    llm = FakeLLM()
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://www.instagram.com/loginwall"})
    assert r.status_code == 422
    data = r.json()
    assert data["success"] is False
    assert data["error"]["code"] == "social_profile_unavailable"
    assert "login" in data["error"]["message"].lower()


def test_social_profile_empty_content_returns_422(client_factory):
    llm = FakeLLM()
    r = client_factory(build_service(llm)).post("/link-analysis", json={"url": "https://www.instagram.com/emptyprofile"})
    assert r.status_code == 422
    data = r.json()
    assert data["success"] is False
    assert data["error"]["code"] == "social_profile_unavailable"
