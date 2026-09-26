"""RecoForge CLI 入口（argparse）。

子命令:
    recoforge benchmark --seed 42 --out benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys

from .core.config import RecoForgeConfig
from .pipeline.pipeline import RecommendationPipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="recoforge", description="RecoForge 推荐系统 CLI")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("benchmark", help="运行评测并输出 benchmark.json")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--n-users", type=int, default=200)
    b.add_argument("--n-items", type=int, default=300)
    b.add_argument("--n-factors", type=int, default=32)
    b.add_argument("--density", type=float, default=0.05)
    b.add_argument("--k", type=int, default=10)
    b.add_argument("--out", type=str, default="benchmark.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd != "benchmark":
        build_parser().print_help()
        return 0

    cfg = RecoForgeConfig.from_env(
        seed=args.seed,
        n_users=args.n_users,
        n_items=args.n_items,
        n_factors=args.n_factors,
        density=args.density,
        k=args.k,
    )
    pipe = RecommendationPipeline(cfg)
    report = pipe.run()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report.__dict__, f, ensure_ascii=False, indent=2)

    print(f"RecoForge benchmark -> {args.out}  (seed={report.seed}, k={report.k})")
    for r in report.results:
        print(
            f"  {r['method']:<14} {r.get('backend', '-'):<8} "
            f"NDCG@K={r['ndcg_at_k']:.4f} Recall@K={r['recall_at_k']:.4f}"
        )
    grade = RecommendationPipeline.grade_report(report)
    print(f"  分级={grade['grade']} ratio={grade['ratio']} ({grade['reason']})")
    return 0 if grade["grade"] in ("S", "A") else 1


if __name__ == "__main__":
    sys.exit(main())
