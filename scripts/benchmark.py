"""Measure latency, token usage and cost for the configured LLM on real websites.

Usage (from the project root, with keys in .env):
    python scripts/benchmark.py https://site1.com https://site2.com https://site3.com
    python scripts/benchmark.py --provider anthropic --model claude-haiku-5-5 \
        --in-price 0.10 --out-price 0.50 https://site1.com https://site2.com

Run it once per candidate model and paste the printed Markdown tables into the README
("Measured results"). Full JSON outputs are saved to benchmark_output/ so you can read them
and score accuracy yourself (rubric in the README): speed and cost are measured,
accuracy is a human judgement.
"""
import argparse
import asyncio
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.errors import AppError  # noqa: E402
from app.services.analysis_service import AnalysisService  # noqa: E402


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("urls", nargs="+")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--in-price", type=float, help="USD per 1M input tokens")
    ap.add_argument("--out-price", type=float, help="USD per 1M output tokens")
    args = ap.parse_args()

    overrides = {}
    if args.provider:
        overrides["llm_provider"] = args.provider
    if args.model:
        overrides["llm_model"] = args.model
    if args.in_price is not None:
        overrides["llm_input_price_per_m"] = args.in_price
    if args.out_price is not None:
        overrides["llm_output_price_per_m"] = args.out_price

    settings = Settings(cache_ttl_seconds=0, **overrides)  # cache off: we want real calls
    service = AnalysisService(settings)
    out_dir = pathlib.Path("benchmark_output")
    out_dir.mkdir(exist_ok=True)

    rows, ok = [], []
    for url in args.urls:
        try:
            r = await service.analyze(url)
        except AppError as exc:
            rows.append((url, "ERROR", exc.code, "", "", ""))
            continue
        m = r.meta
        slug = "".join(c if c.isalnum() else "_" for c in url)[:60]
        (out_dir / f"{m.provider}_{m.model}_{slug}.json").write_text(r.model_dump_json(indent=2))
        cost = f"{m.estimated_cost_usd:.5f}" if m.estimated_cost_usd is not None else "n/a"
        row = (url, f"{m.latency_ms / 1000:.1f}", m.input_tokens, m.output_tokens, cost, len(m.warnings))
        rows.append(row)
        ok.append(row)

    print(f"\n### {settings.llm_provider} / {settings.llm_model}\n")
    print("| Website | Total time (s) | Input tokens | Output tokens | Cost (USD) | Warnings |")
    print("|---|---|---|---|---|---|")
    for row in rows:
        print("| " + " | ".join(str(c) for c in row) + " |")

    if ok:
        print(
            f"\nMedian time: {statistics.median(float(r[1]) for r in ok):.1f}s | "
            f"Avg input tokens: {statistics.mean(r[2] for r in ok):.0f} | "
            f"Avg output tokens: {statistics.mean(r[3] for r in ok):.0f}"
        )


if __name__ == "__main__":
    asyncio.run(main())
