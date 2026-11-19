"""Fetch a website safely and turn its HTML into compact, structured content for the LLM."""
import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from app.config import Settings
from app.errors import AppError, EmptyContentError, WebsiteFetchError
from app.utils.validators import validate_public_url

logger = logging.getLogger(__name__)

PRIORITY_KEYWORDS = (
    "about", "service", "solution", "product", "what-we-do", "offering",
    "platform", "feature", "pricing", "industr", "capabilit", "work",
)
SKIP_EXTENSIONS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".zip",
    ".mp4", ".css", ".js", ".xml", ".ico", ".doc", ".docx",
)


@dataclass
class ParsedPage:
    url: str
    title: str = ""
    meta_description: str = ""
    og_site_name: str = ""
    canonical: str = ""
    lang: str = ""
    headings: list[str] = field(default_factory=list)
    nav_links: list[str] = field(default_factory=list)
    internal_links: list[tuple[str, str]] = field(default_factory=list)  # (text, absolute url)
    jsonld: list[dict] = field(default_factory=list)
    footer_text: str = ""
    text: str = ""
    seo_facts: dict = field(default_factory=dict)


@dataclass
class SiteContent:
    url: str
    final_url: str
    title: str
    meta_description: str
    og_site_name: str
    lang: str
    headings: list[str]
    nav_links: list[str]
    jsonld: list[dict]
    footer_text: str
    seo_facts: dict
    pages: list[dict]  # [{"url","title","text"}]
    warnings: list[str]

    def to_prompt_payload(self) -> dict:
        return {
            "url": self.final_url,
            "title": self.title,
            "meta_description": self.meta_description,
            "site_name": self.og_site_name,
            "language": self.lang,
            "structured_data": self.jsonld,
            "navigation": self.nav_links,
            "headings": self.headings,
            "footer_excerpt": self.footer_text,
            "seo_facts": self.seo_facts,
            "pages": self.pages,
        }


# ------------------------------------------------------------------ parsing
def _clean(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _host_key(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def _truncate_lines(lines: list[str], max_chars: int) -> str:
    out, total = [], 0
    for line in lines:
        if total + len(line) + 1 > max_chars:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(out)


def parse_html(url: str, html: str, max_chars: int = 12_000) -> ParsedPage:
    soup = BeautifulSoup(html, "html.parser")
    page = ParsedPage(url=url)

    page.title = _clean(soup.title.string) if soup.title and soup.title.string else ""

    def meta(attr: str, value: str) -> str:
        tag = soup.find("meta", attrs={attr: re.compile(f"^{re.escape(value)}$", re.I)})
        return _clean(tag.get("content")) if tag and tag.get("content") else ""

    page.meta_description = meta("name", "description") or meta("property", "og:description")
    page.og_site_name = meta("property", "og:site_name")
    canonical = soup.find("link", rel="canonical")
    page.canonical = canonical.get("href", "") if canonical else ""
    html_tag = soup.find("html")
    page.lang = (html_tag.get("lang", "") if html_tag else "")[:10]

    # JSON-LD (Organization / LocalBusiness / Product ...)
    for script in soup.find_all("script", type="application/ld+json")[:10]:
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data.get("@graph", [data]) if isinstance(data, dict) else data if isinstance(data, list) else []
        for item in items:
            if isinstance(item, dict) and item.get("@type") and item.get("name"):
                page.jsonld.append({
                    "type": str(item["@type"])[:60],
                    "name": str(item["name"])[:120],
                    "description": str(item.get("description", ""))[:300],
                })
        page.jsonld = page.jsonld[:5]

    # Headings + on-page SEO facts (measured, not guessed)
    h1s = [_clean(h.get_text()) for h in soup.find_all("h1")]
    h2s = [_clean(h.get_text()) for h in soup.find_all("h2")]
    h3s = [_clean(h.get_text()) for h in soup.find_all("h3")]
    page.headings = [f"H1: {h}" for h in h1s[:3] if h] + [f"H2: {h}" for h in h2s[:12] if h] + [
        f"H3: {h}" for h in h3s[:8] if h
    ]
    images = soup.find_all("img")
    images_total = len(images)
    images_missing_alt = sum(1 for i in images if not _clean(i.get("alt")))

    # Links (collected BEFORE nav/footer are stripped)
    base_host = _host_key(url)
    seen_links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(url, href).split("#")[0]
        parts = urlsplit(absolute)
        if parts.scheme not in ("http", "https") or _host_key(absolute) != base_host:
            continue
        if parts.path.lower().endswith(SKIP_EXTENSIONS) or absolute in seen_links:
            continue
        seen_links.add(absolute)
        page.internal_links.append((_clean(a.get_text())[:60], absolute))

    nav_seen: list[str] = []
    for nav in soup.find_all("nav")[:3]:
        for a in nav.find_all("a"):
            t = _clean(a.get_text())
            if 1 < len(t) <= 40 and t not in nav_seen:
                nav_seen.append(t)
    page.nav_links = nav_seen[:30]

    footer = soup.find("footer")
    page.footer_text = _clean(footer.get_text(" "))[:300] if footer else ""

    # Strip non-content, then extract visible text
    for tag in soup(["script", "style", "noscript", "svg", "iframe", "template", "nav", "footer", "form"]):
        tag.decompose()
    lines, seen = [], set()
    for raw in soup.get_text("\n", strip=True).splitlines():
        line = _clean(raw)
        if len(line) < 3 or line.lower() in seen:
            continue
        seen.add(line.lower())
        lines.append(line)
    page.text = _truncate_lines(lines, max_chars)

    page.seo_facts = {
        "title_length": len(page.title),
        "meta_description_present": bool(page.meta_description),
        "meta_description_length": len(page.meta_description),
        "h1_count": len(h1s),
        "h2_count": len(h2s),
        "images_total": images_total,
        "images_missing_alt": images_missing_alt,
        "has_canonical": bool(page.canonical),
        "has_open_graph_tags": bool(meta("property", "og:title")),
        "has_structured_data": bool(page.jsonld),
        "word_count_visible": len(page.text.split()),
    }
    return page


def pick_priority_links(page: ParsedPage, limit: int) -> list[str]:
    home_path = urlsplit(page.url).path.rstrip("/")
    scored: list[tuple[int, str]] = []
    for text, link in page.internal_links:
        path = urlsplit(link).path.rstrip("/")
        if path == home_path or path.count("/") > 3:
            continue
        haystack = f"{path} {text}".lower()
        score = sum(1 for k in PRIORITY_KEYWORDS if k in haystack)
        if score:
            scored.append((score, link))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    out: list[str] = []
    for _, link in scored:
        if link not in out:
            out.append(link)
        if len(out) >= limit:
            break
    return out


# ------------------------------------------------------------------ fetching
class WebsiteExtractor:
    def __init__(self, settings: Settings, transport: Optional[httpx.AsyncBaseTransport] = None):
        self.settings = settings
        self.transport = transport  # injectable for tests

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.fetch_timeout_seconds),
            headers={
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en;q=0.9",
            },
            follow_redirects=False,  # we follow manually so EVERY hop is SSRF-validated
            trust_env=False,  # don't let proxy env vars bypass our checks
            transport=self.transport,
        )

    async def fetch_html(self, client: httpx.AsyncClient, url: str) -> tuple[str, str]:
        """Return (final_url, html). Validates every redirect hop and caps the body size."""
        current = url
        for _ in range(self.settings.fetch_max_redirects + 1):
            current = await validate_public_url(current)
            try:
                async with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location")
                        if not location:
                            raise WebsiteFetchError("Website sent a redirect without a destination.")
                        current = urljoin(current, location)
                        continue
                    if resp.status_code in (401, 403, 429) or resp.status_code == 503:
                        raise WebsiteFetchError(
                            f"The website refused automated access (HTTP {resp.status_code})."
                        )
                    if resp.status_code >= 400:
                        raise WebsiteFetchError(f"The website returned HTTP {resp.status_code}.")
                    ctype = resp.headers.get("content-type", "").lower()
                    if ctype and not any(t in ctype for t in ("text/html", "application/xhtml", "text/plain")):
                        raise WebsiteFetchError(f"URL did not return an HTML page (content-type: {ctype}).")
                    body = bytearray()
                    async for chunk in resp.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > self.settings.fetch_max_bytes:
                            break  # truncate: the head of the page is what matters
                    encoding = resp.encoding or "utf-8"
                    return current, bytes(body).decode(encoding, errors="replace")
            except httpx.TimeoutException:
                raise WebsiteFetchError("The website took too long to respond.")
            except httpx.TransportError as exc:
                raise WebsiteFetchError(f"Could not connect to the website ({type(exc).__name__}).")
        raise WebsiteFetchError("The website redirected too many times.")

    async def fetch_snippet(self, url: str) -> dict:
        """Lightweight homepage check used to ground competitor info. Raises AppError on failure."""
        safe = await validate_public_url(url)
        async with self._client() as client:
            final_url, html = await self.fetch_html(client, safe)
        page = parse_html(final_url, html, max_chars=1500)
        h1 = next((h[4:] for h in page.headings if h.startswith("H1:")), "")
        return {"url": final_url, "title": page.title, "description": page.meta_description, "h1": h1}

    async def extract(self, raw_url: str) -> SiteContent:
        url = await validate_public_url(raw_url)
        max_chars = self.settings.max_content_chars
        warnings: list[str] = []

        async with self._client() as client:
            final_url, html = await self.fetch_html(client, url)
            home = parse_html(final_url, html, max_chars=max_chars)

            extra_urls = pick_priority_links(home, self.settings.max_extra_pages)

            async def load(link: str) -> Optional[ParsedPage]:
                try:
                    final, page_html = await self.fetch_html(client, link)
                    return parse_html(final, page_html, max_chars=max_chars)
                except AppError as exc:
                    logger.info("Skipping extra page %s: %s", link, exc.message)
                    return None

            extras = [p for p in await asyncio.gather(*(load(u) for u in extra_urls)) if p and p.text]

        home_budget = max_chars if not extras else int(max_chars * 0.6)
        pages = [{"url": home.url, "title": home.title, "text": home.text[:home_budget]}]
        if extras:
            each = (max_chars - home_budget) // len(extras)
            pages += [{"url": p.url, "title": p.title, "text": p.text[:each]} for p in extras]

        total_chars = sum(len(p["text"]) for p in pages)
        if total_chars < 200 and not (home.title or home.meta_description):
            raise EmptyContentError(
                "No readable content found. The site may render everything with JavaScript or block bots."
            )
        if total_chars < 600:
            warnings.append(
                "Very little text was extractable (site may be JavaScript-rendered); analysis may be less specific."
            )

        return SiteContent(
            url=url,
            final_url=final_url,
            title=home.title,
            meta_description=home.meta_description,
            og_site_name=home.og_site_name,
            lang=home.lang,
            headings=home.headings,
            nav_links=home.nav_links,
            jsonld=home.jsonld,
            footer_text=home.footer_text,
            seo_facts=home.seo_facts,
            pages=pages,
            warnings=warnings,
        )
