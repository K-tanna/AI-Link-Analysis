from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, field_validator


# ---------------------------------------------------------------- helpers
def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return value


StrList = Annotated[list[str], BeforeValidator(_as_list)]

PLATFORMS = ("Google", "Facebook", "Instagram", "LinkedIn", "X")
Platform = Literal["Google", "Facebook", "Instagram", "LinkedIn", "X"]

_PLATFORM_ALIASES = {
    "google": "Google",
    "google search": "Google",
    "facebook": "Facebook",
    "instagram": "Instagram",
    "linkedin": "LinkedIn",
    "x": "X",
    "twitter": "X",
    "x (twitter)": "X",
    "x/twitter": "X",
}


def _normalize_platform(value: Any) -> Any:
    if isinstance(value, str):
        return _PLATFORM_ALIASES.get(value.strip().lower(), value)
    return value


class _Model(BaseModel):
    model_config = ConfigDict(extra="ignore")


# ---------------------------------------------------------------- request
class LinkAnalysisRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2048, examples=["https://example.com"])


# ---------------------------------------------------------------- analysis sections
class Brand(_Model):
    name: str = ""
    category: str = ""
    summary: str = ""


class Competitor(_Model):
    name: str
    website: Optional[str] = None
    main_service: str = ""
    target_market: str = ""
    key_strength: str = ""
    strengths: StrList = []
    weaknesses: StrList = []
    # Set by our code, never by the LLM:
    source: Literal["ai_suggestion", "ai_suggestion_homepage_checked"] = "ai_suggestion"


class ComparisonRow(_Model):
    aspect: str
    your_brand: str = ""
    competitors: dict[str, str] = {}


class CompetitorAnalysis(_Model):
    brand_name: str = ""
    brand_overview: str = ""
    main_category: str = ""
    products_services: StrList = []
    competitors: list[Competitor] = []
    comparison: list[ComparisonRow] = []
    strengths: StrList = []  # your brand's competitive advantages
    opportunities: StrList = []  # ways to improve market positioning


class SearchQuery(_Model):
    query: str
    platforms: list[Platform] = Field(min_length=1)
    intent: str = ""
    business_relevance: str = ""
    target_audience: str = ""
    # Always null: we have no real search-volume data source and never invent numbers.
    search_volume: None = None

    @field_validator("platforms", mode="before")
    @classmethod
    def _normalize(cls, v: Any) -> Any:
        v = _as_list(v)
        out: list = []
        for p in v:
            p = _normalize_platform(p)
            if p not in out:
                out.append(p)
        return out


class SupportingContent(_Model):
    type: str = ""  # e.g. landing page, FAQ, blog topic, location page
    title: str = ""
    rationale: str = ""


class SEOImprovements(_Model):
    primary_keyword: str = ""
    secondary_keywords: StrList = []
    long_tail_keywords: StrList = []
    business_keywords: StrList = []
    recommended_title: str = ""
    title_structure: str = ""
    meta_description: str = ""
    current_issues: StrList = []  # grounded in measured on-page facts
    supporting_content: list[SupportingContent] = []


class Service(_Model):
    name: str
    description: str = ""
    target_industry: str = ""
    target_customer: str = ""
    main_benefit: str = ""
    source: Literal["website", "inferred"] = "inferred"


class TargetAudience(_Model):
    primary_audience: StrList = []
    secondary_audience: StrList = []
    industries: StrList = []
    business_types: StrList = []
    customer_profiles: StrList = []
    geographic_focus: StrList = []
    customer_needs: StrList = []
    customer_interests: StrList = []
    recommended_segments: StrList = []


# ---------------------------------------------------------------- LLM stage outputs
class CompetitorCandidate(_Model):
    name: str
    website: Optional[str] = None
    reason: str = ""


class ProfileAndInsights(_Model):
    """Stage 1 LLM output: everything except the competitor comparison."""

    brand: Brand
    brand_overview: str = ""
    main_products_services: StrList = []
    competitor_candidates: list[CompetitorCandidate] = []
    target_search_queries: list[SearchQuery] = Field(min_length=1)
    seo_improvements: SEOImprovements
    top_services: list[Service] = Field(min_length=1)
    target_audience: TargetAudience


# ---------------------------------------------------------------- response
class Analysis(_Model):
    competitor_analysis: CompetitorAnalysis
    target_search_queries: list[SearchQuery]
    seo_improvements: SEOImprovements
    top_services: list[Service]
    target_audience: TargetAudience


class Meta(_Model):
    provider: str
    model: str
    cached: bool = False
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: Optional[float] = None
    pages_analyzed: list[str] = []
    warnings: list[str] = []
    disclaimers: list[str] = []


class LinkAnalysisResponse(_Model):
    success: bool = True
    website: str
    brand: Brand
    analysis: Analysis
    meta: Meta


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Optional[Any] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorBody
