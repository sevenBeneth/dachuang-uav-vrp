#!/usr/bin/env python3
"""10 客户 Solomon 子算例上的标准 NSGA-II 基线。

这是第一阶段的交付内容：一个朴素、教科书式的 NSGA-II（见
``src/uavvrp/nsga2.py``），用于求解 ``src/uavvrp/problem.py`` 中按 split 解码的
VRPTW，不含 ALNS 增强，也不含跨江或 UAV 约束，在小规模算例上运行，并以 Pareto
前沿的形式报告结果。

    # 默认：三个 10 客户算例，distance vs. make-span
    python experiments/run_baseline_n10.py

    # 经典的机队规模目标组合，用于对比
    python experiments/run_baseline_n10.py --objectives distance vehicles

    # 单个算例，更多代数，并逐代输出进度
    python experiments/run_baseline_n10.py --instances RC101_n10 --gen 400 --verbose

目标函数的选择
--------------
默认使用 ``distance`` vs ``makespan``，因为这两个目标确实存在冲突：用一趟紧凑的
巡回服务客户可以把行驶距离压到最小，却会产生一条很长的路径；而把工作拆分到多条
路径上可以平衡各条路径，代价是额外的回仓行程，从而缩短最长的那条路径。

在本解码器下 ``distance`` vs ``vehicles`` **并不**冲突，程序会把这一点明确提示
出来，而不是悄悄返回单点结果。解码器会针对请求的机队规模最小化距离，而向一个可行
方案中增加一辆车永远不会缩短该距离，因此总距离关于车辆数单调不减：车辆数更少在
*两个*目标上都更优，任何更大的机队规模都被支配。在本批子算例上实测，各机队规模
下找到的最小距离为

    C101_n10   k=2:  100.43   k=3:  121.47   k=4:  152.02   k=5:  182.27  ...
    RC101_n10  k=3:  261.04   k=4:  309.05   k=5:  365.84   k=6:  431.36  ...
    R101_n10   k=4:  289.47   k=5:  281.90   k=6:  314.46   k=7:  334.92  ...

只有 R101_n10 在 k=4 与 k=5 之间表现出真正的权衡。这是建模方式本身的性质，而非
搜索失败：要得到有意义的机队规模前沿，需要一种能在距离与车辆数之间做权衡的解码器
（加权和 split，或在 split 目标中加入每辆车的固定成本），这属于后续工作。

每个算例在每种随机种子下求解一次。所报告的前沿是全部种子最终种群的、去重后的
非支配并集 —— 这是报告进化多目标结果的标准做法，因为单次运行只是随机过程的一个
样本。每种随机种子的 hypervolume、前沿规模与最优目标值都会写入 ``runs.csv``，
以便运行之间的离散程度始终保持可见。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uavvrp.nsga2 import NSGA2, fast_non_dominated_sort, hypervolume_2d  # noqa: E402
from uavvrp.problem import OBJECTIVE_NAMES, VRPTWProblem, VRPTWSolution  # noqa: E402
from uavvrp.solomon import Instance  # noqa: E402
from uavvrp.viz import plot_convergence, plot_front_grid, plot_pareto_front  # noqa: E402

DEFAULT_INSTANCES = ("C101_n10", "R101_n10", "RC101_n10")


def load_instance(path: Path) -> tuple[Instance, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    instance = Instance.from_dict(payload)
    return instance, payload


def display_path(path: Path) -> str:
    """用于控制台输出的短路径，尽可能相对于仓库根目录。"""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def merge_fronts(fronts: Sequence[np.ndarray]) -> np.ndarray:
    """多个前沿的非支配并集，按第一个目标排序。"""
    pooled = np.vstack([np.asarray(f, dtype=float) for f in fronts if len(f)])
    if pooled.size == 0:
        return pooled.reshape(0, 0)
    fronts_idx, _ = fast_non_dominated_sort(pooled)
    union = pooled[fronts_idx[0]]
    return union[np.argsort(union[:, 0], kind="stable")]


@dataclass
class InstanceResult:
    name: str
    objectives: tuple[str, ...]
    instance: Instance
    problem: VRPTWProblem
    front: np.ndarray
    front_solutions: list[VRPTWSolution]
    pop_size: int = 0
    n_gen: int = 0
    per_seed: list[dict[str, Any]] = field(default_factory=list)
    histories: list[list[dict]] = field(default_factory=list)
    reference: dict | None = None
    all_points: np.ndarray | None = None
    seconds: float = 0.0


def solve_instance(
    name: str,
    instance: Instance,
    payload: dict,
    *,
    objectives: Sequence[str],
    pop_size: int,
    n_gen: int,
    seeds: Sequence[int],
    verbose: bool,
) -> InstanceResult:
    problem = VRPTWProblem(instance, objectives=objectives)
    started = time.time()

    per_seed: list[dict[str, Any]] = []
    histories: list[list[dict]] = []
    fronts: list[np.ndarray] = []
    all_point_blocks: list[np.ndarray] = []
    pooled_solutions: list[tuple[tuple[float, ...], VRPTWSolution]] = []

    print(
        f"\n[{name}] {instance.n_customers} 个客户，容量 {instance.capacity}，"
        f"车场时间窗 {instance.horizon:g}"
    )
    print(
        f"[{name}] 目标：{' vs '.join(objectives)} | "
        f"pop={pop_size} gen={n_gen} seeds={len(seeds)}"
    )

    for seed in seeds:
        algorithm = NSGA2(
            problem,
            pop_size=pop_size,
            n_gen=n_gen,
            seed=seed,
            hypervolume_reference=problem.nadir,
            verbose=False,
        )
        result = algorithm.run()
        front = result.pareto_front
        fronts.append(front)
        histories.append(result.history)
        all_point_blocks.append(result.objectives)

        hv = (
            hypervolume_2d(front, problem.nadir)
            if front.size and problem.n_obj == 2
            else float("nan")
        )
        for solution, objective_vector in zip(result.pareto_population, front):
            pooled_solutions.append((tuple(objective_vector), solution))

        distinct_front = int(np.unique(np.round(front, 9), axis=0).shape[0]) if front.size else 0
        per_seed.append(
            {
                "seed": seed,
                "front_size": distinct_front,
                "front_members": int(front.shape[0]),
                "hypervolume": hv,
                "best_objectives": front.min(axis=0).tolist() if front.size else [],
                "evaluations": result.n_evaluations,
                "final_generation_history": result.history[-1],
            }
        )

    union = merge_fronts(fronts)
    # 只保留互不相同的目标向量：彼此相等的向量之间不构成支配关系，若不处理，
    # 多个完全相同的前沿求并集时会把同一个解按种子数量重复计入。
    lookup: dict[tuple[float, ...], VRPTWSolution] = {}
    for vector, solution in pooled_solutions:
        lookup.setdefault(tuple(round(v, 9) for v in vector), solution)

    unique_rows: list[np.ndarray] = []
    front_solutions: list[VRPTWSolution] = []
    seen: set[tuple[float, ...]] = set()
    for row in union:
        key = tuple(round(float(v), 9) for v in row)
        if key in seen:
            continue
        solution = lookup.get(key)
        if solution is None:  # pragma: no cover - 防御性检查
            continue
        seen.add(key)
        unique_rows.append(row)
        front_solutions.append(solution)
    front = (
        np.vstack(unique_rows)
        if unique_rows
        else np.zeros((0, union.shape[1] if union.ndim == 2 else 0))
    )

    elapsed = time.time() - started
    hvs = [record["hypervolume"] for record in per_seed]
    print(
        f"[{name}] 并集前沿：{len(front)} 个互不相同的非支配解 | "
        f"hypervolume 中位数 {np.nanmedian(hvs):.4g} "
        f"（最小 {np.nanmin(hvs):.4g}，最大 {np.nanmax(hvs):.4g}）| {elapsed:.1f}s"
    )

    return InstanceResult(
        name=name,
        objectives=tuple(objectives),
        instance=instance,
        problem=problem,
        front=front,
        front_solutions=front_solutions,
        pop_size=pop_size,
        n_gen=n_gen,
        per_seed=per_seed,
        histories=histories,
        reference=payload.get("reference_solution"),
        all_points=np.vstack(all_point_blocks) if all_point_blocks else None,
        seconds=elapsed,
    )


def write_outputs(result: InstanceResult, out_root: Path) -> dict[str, Path]:
    tag = f"{result.name}__{'-'.join(result.objectives)}"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    # ---- Pareto 前沿 CSV --------------------------------------------------
    import pandas as pd

    rows = []
    for rank, solution in enumerate(result.front_solutions):
        record = result.problem.describe(solution)
        rows.append(
            {
                "rank": rank,
                "objective_vector": str([round(v, 6) for v in record["objective_vector"]]),
                "distance": round(record["distance"], 6),
                "vehicles": int(record["vehicles"]),
                "makespan": round(record["makespan"], 6),
                "waiting": round(record["waiting"], 6),
                "n_routes": len(record["routes"]),
                "routes": " | ".join(record["routes"]),
            }
        )
    front_path = out_dir / "pareto_front.csv"
    pd.DataFrame(rows).to_csv(front_path, index=False, encoding="utf-8")
    written["pareto_front"] = front_path

    # ---- 各种子的汇总 -----------------------------------------------------
    seed_rows = []
    for record in result.per_seed:
        seed_rows.append(
            {
                "seed": record["seed"],
                "front_size": record["front_size"],
                "hypervolume": record["hypervolume"],
                "best_objectives": str([round(v, 6) for v in record["best_objectives"]]),
                "evaluations": record["evaluations"],
            }
        )
    runs_path = out_dir / "runs.csv"
    pd.DataFrame(seed_rows).to_csv(runs_path, index=False, encoding="utf-8")
    written["runs"] = runs_path

    # ---- 收敛过程 ---------------------------------------------------------
    convergence_rows = []
    n_generations = len(result.histories[0])
    for generation in range(n_generations):
        hvs = [h[generation].get("hypervolume") for h in result.histories]
        sizes = [h[generation]["front_size"] for h in result.histories]
        convergence_rows.append(
            {
                "generation": generation,
                "hypervolume_median": float(np.nanmedian(hvs)),
                "hypervolume_q1": float(np.nanpercentile(hvs, 25)),
                "hypervolume_q3": float(np.nanpercentile(hvs, 75)),
                "front_size_median": float(np.median(sizes)),
            }
        )
    convergence_path = out_dir / "convergence.csv"
    pd.DataFrame(convergence_rows).to_csv(convergence_path, index=False, encoding="utf-8")
    written["convergence"] = convergence_path

    # ---- 图 ---------------------------------------------------------------
    reference_point = None
    reference_label = "继承的参考解（上界）"
    if result.reference and result.front.size:
        routes = result.reference.get("routes", [])
        reference_point = (
            result.reference["cost"],
            float(len(routes)),
        )
        # 仅当第二个目标是机队规模时才有意义。
        if result.objectives[1] != "vehicles":
            reference_point = None

    front_png = plot_pareto_front(
        result.front,
        result.objectives,
        out_dir / "pareto_front.png",
        title=(
            f"标准 NSGA-II 基线 -- {result.name}\n"
            f"{result.instance.n_customers} 个客户，pop {result.pop_size}，"
            f"{result.n_gen} 代，{len(result.per_seed)} 个种子"
        ),
        reference=reference_point,
        reference_label=reference_label,
        all_points=result.all_points,
    )
    written["pareto_front_png"] = front_png

    convergence_png = plot_convergence(
        result.histories,
        out_dir / "convergence.png",
        title=f"NSGA-II 收敛曲线 -- {result.name}（{len(result.histories)} 个种子的中位数）",
    )
    written["convergence_png"] = convergence_png

    # ---- 机器可读的汇总 ---------------------------------------------------
    summary = {
        "instance": result.name,
        "objectives": list(result.objectives),
        "n_customers": result.instance.n_customers,
        "capacity": result.instance.capacity,
        "depot_horizon": result.instance.horizon,
        "total_demand": result.instance.total_demand,
        "min_vehicles_by_capacity": result.instance.min_vehicles_by_capacity,
        "seeds": [r["seed"] for r in result.per_seed],
        "population_size": result.pop_size,
        "generations": result.n_gen,
        "evaluations_per_run": result.per_seed[0]["evaluations"] if result.per_seed else 0,
        "hypervolume_reference": result.problem.nadir.tolist(),
        "hypervolume_median": float(np.nanmedian([r["hypervolume"] for r in result.per_seed])),
        "hypervolume_min": float(np.nanmin([r["hypervolume"] for r in result.per_seed])),
        "hypervolume_max": float(np.nanmax([r["hypervolume"] for r in result.per_seed])),
        "front_size_union": int(result.front.shape[0]),
        "front_size_median": float(np.median([r["front_size"] for r in result.per_seed])),
        "pareto_front": [[float(v) for v in row] for row in result.front],
        "inherited_reference": {
            "routes": result.reference.get("routes") if result.reference else None,
            "vehicles": result.reference.get("n_vehicles") if result.reference else None,
            "cost": result.reference.get("cost") if result.reference else None,
            "note": "在子算例上有效、但通常非最优的上界",
        },
        "wall_clock_seconds": round(result.seconds, 2),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    written["summary"] = summary_path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--instances", nargs="+", default=list(DEFAULT_INSTANCES),
                        help=f"data/processed/subsets 中的子算例名（默认：{' '.join(DEFAULT_INSTANCES)}）")
    parser.add_argument("--subsets-dir", type=Path,
                        default=ROOT / "data" / "processed" / "subsets")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--objectives", nargs=2, default=["distance", "makespan"],
                        metavar=("OBJ1", "OBJ2"),
                        help=f"要最小化的两个目标，取自 {list(OBJECTIVE_NAMES)} "
                             "（默认：distance makespan）")
    parser.add_argument("--pop", type=int, default=100, help="种群规模（默认 100）")
    parser.add_argument("--gen", type=int, default=200, help="迭代代数（默认 200）")
    parser.add_argument("--seeds", type=int, default=10, help="独立运行次数（默认 10）")
    parser.add_argument("--seed0", type=int, default=1, help="起始随机种子（默认 1）")
    parser.add_argument("--verbose", action="store_true", help="逐代输出进度")
    args = parser.parse_args(argv)

    for objective in args.objectives:
        if objective not in OBJECTIVE_NAMES:
            parser.error(f"未知目标 {objective!r}；请从 {list(OBJECTIVE_NAMES)} 中选择")
    if len(set(args.objectives)) != 2:
        parser.error("两个目标必须互不相同")

    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    panel_data: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []

    print(f"NSGA-II 基线 | 目标 {' vs '.join(args.objectives)} | "
          f"pop={args.pop} gen={args.gen} seeds={len(seeds)}")

    for name in args.instances:
        path = args.subsets_dir / f"{name}.json"
        if not path.exists():
            print(f"缺少子算例 {path}；请先运行 scripts/run_preprocess.py", file=sys.stderr)
            return 2
        instance, payload = load_instance(path)
        result = solve_instance(
            name, instance, payload,
            objectives=args.objectives,
            pop_size=args.pop,
            n_gen=args.gen,
            seeds=seeds,
            verbose=args.verbose,
        )
        written = write_outputs(result, args.out)
        for label, written_path in written.items():
            print(f"[{name}] 已写出 {label}: {display_path(written_path)}")

        panel_data.append(
            {
                "front": result.front,
                "objectives": result.objectives,
                "title": f"{name}  ({instance.n_customers} 个客户)",
                "reference": None,
            }
        )
        overall.append(
            {
                "instance": name,
                "objectives": list(result.objectives),
                "n_customers": instance.n_customers,
                "front_size_union": int(result.front.shape[0]),
                "hypervolume_median": float(np.nanmedian([r["hypervolume"] for r in result.per_seed])),
                "seconds": round(result.seconds, 2),
                "summary_path": display_path(
                    args.out / f"{name}__{'-'.join(args.objectives)}" / "summary.json"
                ),
            }
        )

    if panel_data:
        grid_path = plot_front_grid(
            panel_data,
            args.out / f"pareto_fronts_{'-'.join(args.objectives)}_n10.png",
            title=f"标准 NSGA-II 基线的 Pareto 前沿 "
                  f"({args.objectives[0]} vs {args.objectives[1]}, "
                  f"pop={args.pop}, gen={args.gen}, {len(seeds)} 个种子)",
        )
        print(f"\n已写出合并图：{display_path(grid_path)}")

    print("\n" + "=" * 88)
    print(f"{'算例':<10} {'前沿':>4} {'HV 中位数':>9}  目标")
    print("-" * 88)
    for row in overall:
        print(f"{row['instance']:<12} {row['front_size_union']:>6} "
              f"{row['hypervolume_median']:>12.4g}  {', '.join(row['objectives'])}")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
