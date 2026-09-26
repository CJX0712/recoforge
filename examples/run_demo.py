"""端到端演示：生成 benchmark.json 并校验确定性（同 seed 两次运行逐位一致）。

用法:
    python -m recoforge.examples.run_demo --seed 42 --out benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys

from ..core.config import RecoForgeConfig
from ..pipeline.pipeline import RecommendationPipeline


def _print_table(report) -> None:
    print("\n" + "=" * 64)
    print(f"RecoForge Benchmark  seed={report.seed}  k={report.k}")
    print("=" * 64)
    header = f"{'Method':<14}{'Backend':<9}{'Prec@K':<9}{'Rec@K':<9}{'NDCG@K':<9}{'MAP@K':<9}"
    print(header)
    print("-" * 64)
    for r in report.results:
        print(
            f"{r['method']:<14}"
            f"{r.get('backend', '-'):<9}"
            f"{r['precision_at_k']:<9.4f}"
            f"{r['recall_at_k']:<9.4f}"
            f"{r['ndcg_at_k']:<9.4f}"
            f"{r['map_at_k']:<9.4f}"
        )
    print("-" * 64)
    grade = RecommendationPipeline.grade_report(report)
    print(
        f"性能分级: {grade['grade']}  |  ALS NDCG@K={grade['als_ndcg_at_k']}  "
        f"Pop NDCG@K={grade['pop_ndcg_at_k']}  ratio={grade['ratio']}  ({grade['reason']})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RecoForge demo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-users", type=int, default=200)
    parser.add_argument("--n-items", type=int, default=300)
    parser.add_argument("--n-factors", type=int, default=32)
    parser.add_argument("--density", type=float, default=0.05)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", type=str, default="benchmark.json")
    parser.add_argument("--threshold", type=float, default=1.3)
    args = parser.parse_args(argv)

    cfg = RecoForgeConfig.from_env(
        seed=args.seed,
        n_users=args.n_users,
        n_items=args.n_items,
        n_factors=args.n_factors,
        density=args.density,
        k=args.k,
    )
    pipe = RecommendationPipeline(cfg)

    # 第一次运行
    report1 = pipe.run()
    # 第二次运行：校验确定性（逐位一致）
    report2 = pipe.run()
    _assert_deterministic(report1, report2)

    _print_table(report1)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report1.__dict__, f, ensure_ascii=False, indent=2)
    print(f"\nbenchmark 已写入: {args.out}")

    grade = RecommendationPipeline.grade_report(report1, threshold_ratio=args.threshold)
    # 性能未达 S 时以非零退出，便于 CI 捕获（仍产出报告）
    return 0 if grade["grade"] in ("S", "A") else 1


def _assert_deterministic(r1, r2) -> None:
    for a, b in zip(r1.results, r2.results):
        for key in ("precision_at_k", "recall_at_k", "ndcg_at_k", "map_at_k"):
            if a[key] != b[key]:
                raise RuntimeError(
                    f"确定性校验失败: {a['method']}.{key} {a[key]} != {b[key]}"
                )


if __name__ == "__main__":
    sys.exit(main())
