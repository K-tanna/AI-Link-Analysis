from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlsplit

SOCIAL_HOSTS = {
    "instagram.com": "Instagram", "x.com": "X", "twitter.com": "X",
    "youtube.com": "YouTube", "youtu.be": "YouTube",
    "facebook.com": "Facebook", "fb.com": "Facebook",
    "linkedin.com": "LinkedIn", "tiktok.com": "TikTok",
    "threads.net": "Threads", "threads.com": "Threads",
    "pinterest.com": "Pinterest", "reddit.com": "Reddit",
}

RESERVED_SEGMENTS = {
    "p", "reel", "reels", "explore", "accounts", "stories", "tv", "watch", "hashtag", "search",
    "i", "home", "login", "share", "intent", "status", "pages", "groups", "events", "marketplace",
    "feed", "shorts", "playlist", "results", "channel", "c", "user", "company", "in", "school",
}


@dataclass(frozen=True)
class SocialInfo:
    platform: str
    handle: Optional[str]


def detect_social(url: str) -> Optional[SocialInfo]:
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    platform = None
    for domain, name in SOCIAL_HOSTS.items():
        if host == domain or host.endswith("." + domain):
            platform = name
            break
    if platform is None:
        return None

    segments = [s for s in parts.path.split("/") if s]
    handle: Optional[str] = None
    if segments and host != "youtu.be":
        first = segments[0]
        if platform == "LinkedIn" and first in {"company", "in", "school"} and len(segments) > 1:
            handle = segments[1]
        elif platform == "YouTube" and first in {"c", "channel", "user"} and len(segments) > 1:
            handle = segments[1]
        elif first.lstrip("@").lower() not in RESERVED_SEGMENTS:
            handle = first.lstrip("@")
    return SocialInfo(platform=platform, handle=handle)
