# AI-Powered Website Link Analysis API

FastAPI service that takes a website URL, fetches and cleans the real page content, and uses an LLM to produce five structured sections: **competitor comparison, top-5 search queries, SEO improvements, top-5 services, and target audience**.

- **Endpoint:** `POST http://localhost:4000/link-analysis`
- **Docs (Swagger):** `http://localhost:4000/docs`
- **Stack:** Python 3.12, FastAPI, Uvicorn, Pydantic v2, HTTPX, BeautifulSoup, any of OpenAI / Anthropic / Gemini (plain REST, no vendor SDK)

---

## 1. Setup

```bash
git clone <your-repo-url> && cd ai-link-analysis
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then edit .env and add ONE provider key
uvicorn app.main:app --host 0.0.0.0 --port 4000
```

Minimum `.env` (OpenAI example; for Claude set `LLM_PROVIDER=anthropic`, `LLM_MODEL=claude-haiku-5-5`, `ANTHROPIC_API_KEY=...`):

```
LLM_PROVIDER=openai
LLM_MODEL=gpt-6-luna
OPENAI_API_KEY=sk-...
```

`.env` is git-ignored. No key is ever committed or logged.

### Try it

```bash
curl -X POST http://localhost:4000/link-analysis \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

### Run the tests (offline, no API key needed)

```bash
pytest -q
```

## 2. How it works

```
POST {url}
  │
  ├─ 1. validate_public_url      scheme/port/credentials check + DNS resolve + public-IP check (SSRF)
  ├─ 2. fetch homepage           manual redirects (each hop re-validated), 10 s timeout, 2 MB cap, HTML only
  ├─ 3. fetch ≤2 key pages       /about, /services, /products… picked by link keywords
  ├─ 4. clean + structure        title, meta, headings, nav, JSON-LD, footer, measured on-page SEO facts,
  │                              visible text (scripts/styles/nav/forms stripped, de-duplicated, ≤12k chars)
  ├─ 5. LLM call 1  (profile)    brand, 5 services, 5 queries, SEO, audience, competitor *candidates*
  ├─ 6. competitor grounding     fetch each candidate's homepage (no LLM) → real title/description as evidence
  ├─ 7. LLM call 2  (compare)    comparison written against that evidence
  └─ 8. Pydantic validation      one repair retry on bad/truncated JSON, then 502 · result cached 1 h
```

Why two calls instead of one: the comparison step can only be grounded *after* we know who the competitors are and have looked at their sites. It also keeps each prompt short and the output JSON small enough to avoid truncation.

### Design decisions that follow the brief

| Brief requirement | How it is handled |
|---|---|
| Use actual website content | Prompt contains only extracted page data; system prompt forbids inventing products/clients/locations. Services carry `source: "website" \| "inferred"`. |
| Competitors credible or clearly labelled | Competitors are AI suggestions; each is labelled `ai_suggestion` or `ai_suggestion_homepage_checked` (**set by code, never by the LLM**). The brand's own domain is removed from the list. |
| Don't invent search volumes | `search_volume` is always `null`; `meta.disclaimers` says why. |
| SEO grounded in the page | Measured facts (title/meta length, H1 count, images missing alt, canonical, OG, JSON-LD) are passed to the model and it must base `current_issues` on them. |
| SSRF protection | See §4. |
| Graceful failures | Structured error JSON for every failure path (§3). |
| Prompt-injection | Page text is wrapped as data; system prompt tells the model to ignore instructions inside it. |

## 3. API

### Request

```json
{ "url": "https://example.com" }
```
`example.com` (no scheme) is also accepted and treated as `https://`.

### Response (shape)

> **Illustrative, trimmed.** This shows the structure. It is *not* captured LLM output. See "Sample output" at the end of this section for how to add a real one.

```json
{
  "success": true,
  "website": "https://example.com/",
  "brand": { "name": "Acme Web Studio", "category": "Web Development", "summary": "Builds websites for small businesses." },
  "analysis": {
    "competitor_analysis": {
      "brand_name": "Acme Web Studio",
      "brand_overview": "...",
      "main_category": "Web Development",
      "products_services": ["Web design", "E-commerce"],
      "competitors": [
        { "name": "Rival One", "website": "https://rival-one.com/", "main_service": "...", "target_market": "...",
          "key_strength": "...", "strengths": ["..."], "weaknesses": ["..."],
          "source": "ai_suggestion_homepage_checked" }
      ],
      "comparison": [
        { "aspect": "Target Market", "your_brand": "Small businesses", "competitors": { "Rival One": "Enterprise" } }
      ],
      "strengths": ["..."],
      "opportunities": ["..."]
    },
    "target_search_queries": [
      { "query": "affordable website design Ahmedabad", "platforms": ["Google"], "intent": "Service Search",
        "business_relevance": "...", "target_audience": "...", "search_volume": null }
    ],
    "seo_improvements": {
      "primary_keyword": "...", "secondary_keywords": [], "long_tail_keywords": [], "business_keywords": [],
      "recommended_title": "...", "title_structure": "...", "meta_description": "...",
      "current_issues": ["..."],
      "supporting_content": [ { "type": "FAQ", "title": "...", "rationale": "..." } ]
    },
    "top_services": [
      { "name": "...", "description": "...", "target_industry": "...", "target_customer": "...",
        "main_benefit": "...", "source": "website" }
    ],
    "target_audience": {
      "primary_audience": [], "secondary_audience": [], "industries": [], "business_types": [],
      "customer_profiles": [], "geographic_focus": [], "customer_needs": [], "customer_interests": [],
      "recommended_segments": []
    }
  },
  "meta": {
    "provider": "openai", "model": "gpt-6-luna", "cached": false, "latency_ms": 0,
    "input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0,
    "pages_analyzed": ["https://example.com/", "https://example.com/about"],
    "warnings": [], "disclaimers": ["..."]
  }
}
```

### Errors

All errors share one shape: `{"success": false, "error": {"code": "...", "message": "...", "details": null}}`

| HTTP | `code` | When |
|---|---|---|
| 400 | `invalid_url` | Malformed URL, bad scheme, unresolvable domain, credentials in URL |
| 400 | `blocked_url` | Private/loopback/link-local/metadata address, non-80/443 port, or a redirect to one |
| 422 | `invalid_request` | Body missing `url` or wrong type |
| 422 | `website_unreachable` | Timeout, connection error, HTTP 4xx/5xx, bot-blocking (403/429/503), non-HTML |
| 422 | `no_extractable_content` | Page has no readable text (typically JavaScript-only rendering) |
| 502 | `llm_unavailable` | Provider down / auth failed after retries (and fallback, if configured) |
| 502 | `llm_invalid_output` | Model returned invalid JSON/structure twice |
| 503 | `llm_not_configured` | No API key for the selected provider |

### Sample output

Run a real analysis and paste it here (this is a required submission item):

```bash
curl -s -X POST localhost:4000/link-analysis -H "Content-Type: application/json" \
  -d '{"url":"https://<a-real-site>"}' | python -m json.tool > sample_response.json
```

## 4. Security

- **SSRF:** every URL fetched (submitted URL, each redirect hop, each extra page, each competitor homepage) is resolved and rejected if *any* address is loopback, RFC 1918, link-local (169.254.169.254 cloud metadata), CGNAT, reserved, multicast or IPv4-mapped IPv6 of those. Only ports 80/443. Redirects are followed manually so each hop is re-checked. `trust_env=False` stops proxy env vars bypassing the checks.
- **Resource limits:** 10 s timeout, 2 MB body cap, max 5 redirects, HTML content types only.
- **Secrets:** read from environment only; `.env` git-ignored; provider error bodies for 401/403 are not echoed back to clients.
- **Known limitation:** a DNS-rebinding race exists between our DNS check and the socket connect. In production, add an egress proxy or network policy that blocks private ranges.

## 5. Test and run logs

```
$ pytest -q
.......................................................                  [100%]
55 passed
```

Covers: SSRF (loopback, RFC1918, metadata IP, IPv6, IPv4-mapped IPv6, CGNAT, private-resolving hostname, redirect-to-private), URL validation, HTML extraction, each provider's response/usage parsing, retries, fallback, and the full endpoint including every error path and cache behaviour.

Smoke test against the running server (sandbox had no LLM key / outbound DNS):

```
GET /docs               -> 200
GET /health             -> {"status":"ok"}
POST {"url":"http://127.0.0.1:4000/health"}
  -> {"success":false,"error":{"code":"blocked_url","message":"Only ports 80 and 443 are allowed.",...}}
POST {"url":"http://169.254.169.254/latest/meta-data/"}
  -> {"success":false,"error":{"code":"blocked_url","message":"This URL points to a private, loopback or otherwise non-public address...",...}}
POST {"url":"not a url"}   -> 400 invalid_url
POST {}                    -> 422 invalid_request
```

> **Add before submitting:** a screenshot of `/docs` and a screenshot or log of a **real** analysis (including the `meta` block with tokens/latency/cost).

---

# 6. LLM Platform and Model Research

**Prices verified 18 Nov 2026** from provider pricing pages and release coverage (links at the end). Prices change; re-check before relying on them.

## A. Comparison

| | **GPT-6 Luna** | **Claude Haiku 5.5** | **Gemini 3.8 Flash** | *Claude Sonnet 5.5 (quality reference)* |
|---|---|---|---|---|
| Provider | OpenAI | Anthropic | Google | Anthropic |
| Model ID | `gpt-6-luna` | `claude-haiku-5-5` | `gemini-3.8-flash` (verify id) | `claude-sonnet-5-5` |
| Input / 1M tokens | $0.10 | $0.10 (prompt ≤100K), $0.50 above | $0.75 *introductory to 31 Dec 2026* | $2.00 |
| Output / 1M tokens | $0.50 | $0.50 (≤100K), $2.50 above | $3.75 *introductory* | $10.00 |
| Context window | 1,050,000 | 1,000,000 | 1,048,576 | not verified |
| JSON support | Native structured outputs (documented) | JSON via prompt + our validation/repair; native schema mode not verified | JSON mode (`responseMimeType`) | as Haiku |
| Response speed | **not measured** | **not measured** | **not measured** | **not measured** |
| Accuracy on this task | **not tested** | **not tested** | **not tested** | **not tested** |
| Fit | Primary candidate | Fallback candidate | Works, ~7.5× the cost | Only if cheaper models fail the rubric |

Notes: our prompts are ~5K tokens, far below Haiku's 100K price cliff, so that cliff doesn't affect us. Third-party trackers disagree on Gemini 3.8 Flash's price (one lists half of Google's figure); I used Google's own Cloud pricing page. Gemini's introductory price ends 31 Dec 2026; the post-intro price for 3.8 Flash is not something I verified, and Google announced $1.50 / $7.50 for the sibling 3.7 Flash from 1 Jan 2027.

## B. Selection criteria → verdict

1. **Accuracy:** unproven for all models until the benchmark below is run.
2. **Speed:** unproven; output tokens dominate latency, so a model that writes fewer tokens is usually faster.
3. **Cost:** Luna and Haiku 5.5 are tied on list price and ~7.5× cheaper than Gemini 3.8 Flash.
4. **Reliable JSON:** Luna has documented native structured outputs; the others rely on JSON mode or prompting. Our validate-and-repair loop covers all three.
5. **Competitor/SEO insight:** depends on model knowledge; needs human scoring.
6. **Large content:** all have ~1M context; we send ~3K tokens of page text, so this isn't a differentiator.

## C. Cost optimisation

**Already implemented**
- Strip scripts, styles, nav, footer, forms; de-duplicate lines; hard cap of 12,000 chars of page text.
- Only 2 extra pages, chosen by keyword relevance, with a shared character budget.
- Compact JSON payload (no pretty-printing); compact schema in the prompt.
- Competitor evidence gathered by plain HTTP (**zero LLM tokens**).
- Two focused calls instead of one giant call, so outputs stay small and rarely truncate.
- At most **one** repair retry, so a misbehaving model can't loop and burn tokens.
- 1-hour in-memory cache keyed by normalised URL (the SSRF check still runs on cache hits).
- Cheapest viable model class by default; fallback only on provider failure.

**Recommended next (not implemented)**
- Batch APIs for bulk jobs: Haiku 5.5 batch list price is $0.05 / $0.25 per 1M (half of standard).
- Prompt caching for the static system prompt + schema (cache reads cost $0.01/1M on Haiku 5.5 and Luna). Our static prefix is small, so check the provider's minimum cacheable size before expecting savings.
- Model routing: cheap model for extraction/classification, larger model only for synthesis if scores justify it.
- Redis/shared cache when running multiple instances.

## D. Final recommendation

| | |
|---|---|
| **Primary** | OpenAI **`gpt-6-luna`** |
| **Fallback** | Anthropic **`claude-haiku-5-5`** (different provider → survives an OpenAI outage; same price) |
| Est. input tokens / analysis | ~7,000 (call 1 ≈ 5K, call 2 ≈ 2K) — *design estimate* |
| Est. output tokens / analysis | ~3,500 (call 1 ≈ 2.5K, call 2 ≈ 1K) — *design estimate* |
| **Est. cost / analysis** | **≈ $0.0024** (≈ $2.45 per 1,000 sites) — *estimate from list prices* |
| Expected response time | **Not measured.** Rough expectation: tens of seconds end-to-end, dominated by generating ~3.5K output tokens across two sequential calls. Replace with benchmark numbers. |

**Why Luna first:** (1) lowest list price tier; (2) documented native structured-output support, which matters because the whole API is "reliable JSON"; (3) independent testing reported by Artificial Analysis (via press coverage) found Haiku 5.5 used about **3× as many output tokens per task** as Luna. Output is the expensive side, so if that holds for our prompts, Haiku's real cost would be ≈ **$0.0060** per analysis (≈ $5.95 / 1,000 sites) rather than $0.0024. That is a reported benchmark on other tasks, **not something I tested on this workload.**

**Cost comparison (estimate: 7,000 in / 3,500 out)**

| Model | Per analysis | Per 1,000 analyses |
|---|---|---|
| GPT-6 Luna | $0.0024 | $2.45 |
| Claude Haiku 5.5 | $0.0024 | $2.45 |
| Claude Haiku 5.5, if it writes 3× the output | $0.0060 | $5.95 |
| Gemini 3.8 Flash (intro price) | $0.0184 | $18.38 |
| Claude Sonnet 5.5 | $0.0490 | $49.00 |

**Balancing approach:** run the cheap primary, validate output strictly, fail over on provider errors, and escalate to a larger model only if the accuracy rubric below shows the cheap models are inadequate. I'm deliberately not claiming "maximum accuracy": that requires testing.

### How to turn estimates into measurements

```bash
# one run per candidate model, on the same 5 websites of different types
python scripts/benchmark.py --provider openai    --model gpt-6-luna       --in-price 0.10 --out-price 0.50 URL1 URL2 URL3 URL4 URL5
python scripts/benchmark.py --provider anthropic --model claude-haiku-5-5 --in-price 0.10 --out-price 0.50 URL1 URL2 URL3 URL4 URL5
python scripts/benchmark.py --provider gemini    --model gemini-3.8-flash --in-price 0.75 --out-price 3.75 URL1 URL2 URL3 URL4 URL5
```

It prints Markdown tables (time, tokens, cost) and saves each full output to `benchmark_output/`.

**Accuracy rubric (score each output 0–2, by hand):** brand/category correct · services actually on the site · queries plausible for real customers · competitors real and relevant · SEO advice specific to this site · audience matches the site · valid JSON without repair.

### Measured results

> *Paste the benchmark tables here once you've run them. Until then, every speed and accuracy cell above is "not measured".*

---

## 7. Project structure

```
app/
  main.py                      FastAPI app, endpoint, error handlers
  config.py                    env-based settings
  schemas.py                   request/response + LLM-output models
  errors.py                    typed errors → HTTP status/code
  services/
    website_extractor.py       safe fetch + HTML → structured content
    llm_service.py             OpenAI/Anthropic/Gemini client, retries, fallback
    analysis_service.py        prompts, 2-stage pipeline, validation, cache
    competitor_service.py      competitor homepage grounding
  utils/validators.py          URL validation + SSRF protection
scripts/benchmark.py           measure latency/tokens/cost per model
tests/test_link_analysis.py    55 offline tests
```

## 8. Known limitations

- JavaScript-rendered sites return little text (we detect this and warn or return `no_extractable_content`). A headless browser (Playwright) would fix it at the cost of speed and infrastructure.
- Competitors are LLM suggestions; the homepage check proves a domain exists and gives context, not that the competitive claims are true.
- Some sites block bots (403/429); we report that clearly rather than working around it.
- In-memory cache is per-process.
- Speed and accuracy figures are unmeasured until you run the benchmark.

## Sources (pricing/specs, checked 18 Nov 2026)

- OpenAI model page for GPT-6 Luna (context, structured outputs, $0.10 input): developers.openai.com/api/docs/models/gpt-6-luna
- Claude Haiku 5.5 pricing, 100K tier, batch and cache rates: platform.claude.com pricing docs, via ppc.land, mixed-news.com, developersdigest.tech
- Haiku 5.5 vs Luna output-token usage (Artificial Analysis): implicator.ai coverage
- Gemini 3.8 Flash introductory pricing: cloud.google.com/vertex-ai/generative-ai/pricing; benchlm.ai; context window from pricepertoken.com
- Claude Sonnet 5.5 price: mixed-news.com
