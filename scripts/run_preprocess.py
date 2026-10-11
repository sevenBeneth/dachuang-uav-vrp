#!/usr/bin/env python3
"""运行 Solomon 预处理流水线。

    python scripts/run_preprocess.py
    python scripts/run_preprocess.py --subset C101:10 --subset R101:10 --subset RC101:10

读取 ``data/raw/solomon``，并将校验、归一化后的 JSON 算例、固定规模子算例以及
校验报告写入 ``data/processed``。任一校验失败则返回非零退出码。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavvrp.preprocess import DEFAULT_SUBSETS, preprocess  # noqa: E402


def _parse_subset(spec: str) -> tuple[str, int]:
    name, sep, size = spec.partition(":")
    if not sep or not name or not size.isdigit():
        raise argparse.ArgumentTypeError(f"期望 NAME:SIZE，实际收到 {spec!r}")
    return name.upper(), int(size)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "data" / "raw" / "solomon",
        help="存放原始 .vrp/.sol 文件的目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "processed",
        help="归一化算例与报告的输出目录",
    )
    parser.add_argument(
        "--subset",
        type=_parse_subset,
        action="append",
        default=None,
        metavar="NAME:SIZE",
        help=f"要抽取的子算例（可重复指定）。默认：{DEFAULT_SUBSETS}",
    )
    parser.add_argument("--quiet", action="store_true", help="不输出进度信息")
    args = parser.parse_args(argv)

    subsets = tuple(args.subset) if args.subset else DEFAULT_SUBSETS
    report = preprocess(args.raw, args.out, subsets=subsets, verbose=not args.quiet)

    print()
    print("=" * 78)
    print(f"{'算例':<8} {'族':<6} {'客户数':>3} {'容量':>3} {'车辆上限':>4} "
          f"{'参考车辆':>4} {'参考成本':>6} {'重算成本':>7}")
    print("-" * 78)
    for name in sorted(report["instances"]):
        row = report["instances"][name]
        print(f"{name:<10} {name[:2].rstrip('0123456789'):<7} {row['n_customers']:>6} "
              f"{row['capacity']:>5} {row['vehicle_limit']:>8} "
              f"{str(row['ref_vehicles']):>8} {str(row['ref_cost_published']):>10} "
              f"{str(row['ref_cost_recomputed']):>11}")

    if report["subsets"]:
        print()
        print("NSGA-II 基线所用的子算例")
        print("-" * 78)
        print(f"{'名称':<10} {'客户数':>3} {'总需求':>5} {'容量':>3} "
              f"{'参考车辆':>4} {'参考距离':>6} {'校验':>6}")
        for name, row in sorted(report["subsets"].items()):
            print(f"{name:<12} {row['n_customers']:>6} {row['total_demand']:>8} "
                  f"{row['capacity']:>5} {str(row['reference_vehicles']):>8} "
                  f"{str(row['reference_distance']):>10} "
                  f"{row['checks_passed']}/{row['checks_total']:>3}")

    summary = report["summary"]
    print()
    print(f"算例数={summary['instances']}  校验数={summary['checks_total']}  "
          f"错误数={summary['errors']}  提示数={summary['advisories']}  "
          f"子算例数={summary['subsets']}")
    if report["advisories"]:
        print(f"\n上游基准算例上的 ADVISORY 提示（{len(report['advisories'])} 个算例）"
              " -- 详见 docs/data_notes.md：")
        for entry in report["advisories"][:6]:
            for check in entry["advisories"]:
                print(f"  {entry['name']:<8} {check['check']}: {check['detail']}")
        if len(report["advisories"]) > 6:
            print(f"  ... 另有 {len(report['advisories']) - 6} 个（完整列表见 validation_report.json）")
    if report["failures"]:
        print("\n校验失败：", file=sys.stderr)
        for failure in report["failures"]:
            print(f"  {failure['name']}", file=sys.stderr)
            for check in failure["failed"]:
                print(f"     - {check['check']}: {check['detail']}", file=sys.stderr)
        return 1
    print("\n所有结构校验均已通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
