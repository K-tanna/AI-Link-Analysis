"""Grounds LLM-suggested competitors with a real homepage check.

The LLM proposes competitors from general market knowledge. That is NOT verified research, so
we (a) fetch each suggested competitor's homepage (SSRF-validated) to confirm the domain exists
and to give the comparison step real title/description text, and (b) label every competitor with
a `source` so consumers can tell checked suggestions from unchecked ones.
"""
import asyncio
import logging
from urllib.parse import urlsplit

from app.errors import AppError
from app.services.website_extractor import WebsiteExtractor
from app.utils.validators import normalize_url

logger = logging.getLogger(__name__)

MAX_COMPETITORS = 5


def _host_key(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


class CompetitorService:
    def __init__(self, extractor: WebsiteExtractor):
        self.extractor = extractor

    async def build_context(self, candidates: list[dict], own_url: str) -> list[dict]:
        """Return de-duplicated candidates enriched with homepage evidence where reachable."""
        own_host = _host_key(own_url)
        seen: set[str] = set()
        selected: list[dict] = []

        for cand in candidates:
            name = (cand.get("name") or "").strip()
            if not name or name.lower() in seen:
                continue
            website = None
            if cand.get("website"):
                try:
                    website = normalize_url(cand["website"])
                except AppError:
                    website = None
            if website and _host_key(website) == own_host:
                continue  # the brand itself is not its own competitor
            seen.add(name.lower())
            selected.append({"name": name, "website": website, "reason": (cand.get("reason") or "")[:200]})
            if len(selected) >= MAX_COMPETITORS:
                break

        async def check(item: dict) -> dict:
            item = dict(item, homepage_checked=False, homepage_title="", homepage_description="")
            if not item["website"]:
                return item
            try:
                snippet = await self.extractor.fetch_snippet(item["website"])
                item.update(
                    homepage_checked=True,
                    homepage_title=snippet["title"][:150],
                    homepage_description=(snippet["description"] or snippet["h1"])[:300],
                )
            except AppError as exc:
                logger.info("Competitor homepage check failed for %s: %s", item["website"], exc.message)
            return item

        return list(await asyncio.gather(*(check(i) for i in selected)))
