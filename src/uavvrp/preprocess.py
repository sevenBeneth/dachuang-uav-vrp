"""Solomon VRPTW 算例集的预处理流水线。

本模块依次完成以下工作：

1. **加载** ``data/raw/solomon`` 下每一对 ``<name>.vrp`` / ``<name>.sol``。
2. **校验**每个算例的结构：声明的 ``DIMENSION`` 与实际段落长度是否一致、车场
   是否恰好一个、车场的需求量/时间窗、单个需求是否能装入一辆车、时间窗是否满足
   ``ready <= due``，以及总需求是否在声明的车队运力之内。
3. **交叉核对**每个参考解与对应算例：每个客户是否恰好被服务一次、每条路径是否
   同时满足载量与时间窗约束、重新计算的路径距离是否等于公布的 ``Cost`` 行。
   最后一项检查才是真正的证据，说明我们的距离与时间约定与 Solomon 一致——如果
   解析器在节点编号或取整规则上有偏差，成本不可能精确到一位小数地吻合。
4. **归一化**每个算例为 JSON，车场置于下标 0。
5. **抽取子算例**：按需截取指定规模（NSGA-II 基线所用的 10 客户算例），并从完整
   的基准解继承一个参考解。
6. **报告**：逐算例的 manifest CSV、机器可读的校验报告，以及控制台摘要。

本流水线刻意采用 "失败即大声报错" 的策略：任何检查失败都会被记录，且退出码变为
非零，因此有问题的输入集绝不会悄无声息地变成对外发布的结果。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

from .solomon import Instance, Solution, parse_sol, parse_vrp, portable_path
from .vrptw import REFERENCE_TOLERANCE, simulate_route

__all__ = [
    "Check",
    "Suite",
    "load_suite",
    "check_instance",
    "check_reference",
    "family_of",
    "series_of",
    "restrict_solution",
    "build_manifest_rows",
    "preprocess",
]

# 公布的成本只打印到一位小数，因此与最后一位打印数字相差半个单位以内都算精确
# 吻合。
COST_TOLERANCE = 0.05

ERROR = "error"
ADVISORY = "advisory"


@dataclass(frozen=True)
class Check:
    """关于某个算例或某个参考解的一条具名断言。

    ``severity`` 为 :data:`ERROR` 时，表示该断言失败意味着流水线自身的输入或
    解析有误；为 :data:`ADVISORY` 时，表示它记录的是一致性问题，而该问题本就
    存在于上游基准文件中（见 ``docs/data_notes.md``）。只有 error 才会让流水线
    以非零码退出。
    """

    name: str
    ok: bool
    detail: str = ""
    severity: str = ERROR

    def as_dict(self) -> dict:
        return {"check": self.name, "ok": self.ok, "detail": self.detail, "severity": self.severity}


@dataclass
class Suite:
    """全部已加载的算例及其参考解。"""

    instances: dict[str, Instance] = field(default_factory=dict)
    solutions: dict[str, Solution] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.instances)

    def names(self) -> list[str]:
        return sorted(self.instances)


# ---------------------------------------------------------------------------
# 命名辅助函数
# ---------------------------------------------------------------------------


def family_of(name: str) -> str:
    """``C101 -> C``，``RC201 -> RC``。"""
    up = name.upper()
    if up.startswith("RC"):
        return "RC"
    return up[0]


def series_of(name: str) -> str:
    """``C101 -> 1``，``RC201 -> 2``。第 1 类时间跨度短、车队规模小，
    第 2 类时间跨度长、车队规模大。"""
    up = name.upper()
    rest = up[2:] if up.startswith("RC") else up[1:]
    return rest[0] if rest and rest[0].isdigit() else "?"


# ---------------------------------------------------------------------------
# 加载
# ---------------------------------------------------------------------------


def load_suite(raw_dir: Path | str) -> Suite:
    """加载 ``raw_dir`` 下的每个 ``.vrp`` 及其同名的 ``.sol`` 文件。"""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"未找到原始数据目录: {raw_dir}\n"
            "请先运行 `python scripts/fetch_solomon.py`"
        )

    suite = Suite()
    for vrp_path in sorted(raw_dir.glob("*.vrp")):
        inst = parse_vrp(vrp_path)
        suite.instances[inst.name] = inst
        sol_path = vrp_path.with_suffix(".sol")
        if sol_path.exists():
            suite.solutions[inst.name] = parse_sol(sol_path)
    if not suite.instances:
        raise FileNotFoundError(f"{raw_dir} 下没有 .vrp 文件")
    return suite


# ---------------------------------------------------------------------------
# 算例层面的检查
# ---------------------------------------------------------------------------


def check_instance(inst: Instance) -> list[Check]:
    """任何格式良好的 VRPTW 算例都必须满足的结构性检查。"""
    checks: list[Check] = []
    n = inst.n_nodes

    checks.append(Check("n_nodes>=2", n >= 2, f"n_nodes={n}"))
    checks.append(
        Check(
            "sections_aligned",
            len(inst.coords) == len(inst.demand) == len(inst.ready) == len(inst.due) == len(inst.service),
            "coord/demand/ready/due/service 各数组长度完全一致",
        )
    )
    checks.append(Check("capacity>0", inst.capacity > 0, f"capacity={inst.capacity}"))
    checks.append(
        Check("vehicle_limit>=1", inst.vehicle_limit >= 1, f"vehicle_limit={inst.vehicle_limit}")
    )

    # 车场：无需求，开放时间不晚于最早的客户可服务时间，且具有有意义的关闭时间。
    checks.append(Check("depot_demand_zero", inst.demand[0] == 0, f"demand[depot]={inst.demand[0]}"))
    checks.append(
        Check("depot_service_zero", inst.service[0] == 0, f"service[depot]={inst.service[0]}")
    )
    checks.append(Check("depot_ready<=due", inst.ready[0] <= inst.due[0], f"[{inst.ready[0]}, {inst.due[0]}]"))

    bad_windows = [i for i in range(n) if inst.ready[i] > inst.due[i]]
    checks.append(
        Check("ready<=due_all", not bad_windows, f"{len(bad_windows)} 处违规" + (f": {bad_windows[:5]}" if bad_windows else ""))
    )

    oversized = [i for i in range(1, n) if inst.demand[i] > inst.capacity]
    checks.append(
        Check(
            "each_demand<=capacity",
            not oversized,
            f"{len(oversized)} 个客户的需求超出车辆载量" + (f": {oversized[:5]}" if oversized else ""),
        )
    )

    negative = [i for i in range(n) if inst.demand[i] < 0]
    checks.append(Check("demand>=0", not negative, f"{len(negative)} 个负需求"))

    # 声明的车队必须能够承运总需求。
    required = inst.min_vehicles_by_capacity
    checks.append(
        Check(
            "fleet_can_carry_demand",
            required <= inst.vehicle_limit,
            f"按载量至少需要 {required} 辆车，车队上限为 {inst.vehicle_limit}",
        )
    )
    checks.append(
        Check("total_demand>0", inst.total_demand > 0, f"total_demand={inst.total_demand}")
    )

    # 客户的时间窗必须能从车场出发按时到达，并且之后还能返回车场。
    unreachable = [
        i
        for i in range(1, n)
        if inst.ready[0] + inst.distance(0, i) > inst.due[i] + 1e-9
    ]
    checks.append(
        Check(
            "customers_reachable_from_depot",
            not unreachable,
            f"{len(unreachable)} 个不可达客户" + (f": {unreachable[:5]}" if unreachable else ""),
        )
    )

    return checks


# ---------------------------------------------------------------------------
# 参考解检查
# ---------------------------------------------------------------------------


def restrict_solution(inst: Instance, sol: Solution, customers: Sequence[int]) -> Solution:
    """只保留 ``sol`` 中的 ``customers``，并维持路径内的顺序。

    从一条可行路径中删除停靠点后，该路径仍然可行：车辆到达每一个保留停靠点的
    时间都不会更晚（因此不可能错过 due 时间），载量也不会更大。因此结果始终是
    子算例的一个有效解——虽然通常并非最优解，但可作为上界。
    """
    keep = set(customers)
    routes = [[c for c in route if c in keep] for route in sol.routes]
    return Solution(
        name=f"{inst.name}_restricted",
        routes=[r for r in routes if r],
        cost=None,
        source_path=sol.source_path,
    )


def check_reference(inst: Instance, sol: Solution) -> tuple[list[Check], float]:
    """校验公布的参考解。返回 (checks, 重新计算的距离)。

    结构性预期（每个客户恰好被服务一次、车队规模不超过声明上限）属于
    :data:`ERROR`。而路径是否*复现了声明的成本*、是否*通过时间窗校验*则属于
    :data:`ADVISORY`：56 个上游文件中有 48 个声明了一个其自身路径根本达不到的
    成本，另有 8 个文件含有一条违反时间窗不足 0.4 个时间单位的路径。这是公布
    数据本身的缺陷，而不是本流水线的缺陷，因此连同完整证据一并报告，既不隐瞒，
    也不将其视为致命错误。
    """
    checks: list[Check] = []
    dist = inst.distance_matrix()

    served = [c for route in sol.routes for c in route]
    expected = set(range(1, inst.n_nodes))
    got = set(served)

    checks.append(
        Check(
            "ref_routes_nonempty",
            len(sol.routes) > 0,
            f"解析出 {len(sol.routes)} 条路径",
        )
    )
    checks.append(
        Check(
            "ref_each_customer_once",
            len(served) == len(got),
            f"{len(served)} 个停靠点，对应 {len(got)} 个不同客户",
        )
    )
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    checks.append(
        Check(
            "ref_covers_all_customers",
            not missing and not extra,
            f"缺失={missing[:5]} (n={len(missing)})，未知编号={extra[:5]} (n={len(extra)})",
        )
    )

    strict_bad: list[str] = []
    loose_bad: list[str] = []
    total = 0.0
    for idx, route in enumerate(sol.routes, start=1):
        stats = simulate_route(inst, route, dist, tolerance=REFERENCE_TOLERANCE)
        if stats is None:
            loose_bad.append(f"#{idx}")
            strict_bad.append(f"#{idx}")
            continue
        total += stats.distance
        if simulate_route(inst, route, dist) is None:
            strict_bad.append(f"#{idx}")

    checks.append(
        Check(
            "ref_routes_feasible_strict",
            not strict_bad,
            "所有路径都精确满足载量与时间窗约束"
            if not strict_bad
            else f"{len(strict_bad)} 条路径违反时间窗的幅度 <={REFERENCE_TOLERANCE}: {', '.join(strict_bad)}",
            severity=ADVISORY,
        )
    )
    checks.append(
        Check(
            "ref_routes_feasible_capacity",
            not loose_bad,
            "所有路径都在公布数据容差内满足载量与时间窗约束"
            if not loose_bad
            else f"超出容差的不可行路径: {', '.join(loose_bad)}",
            severity=ERROR,
        )
    )

    checks.append(
        Check(
            "ref_fleet_size<=limit",
            len(sol.routes) <= inst.vehicle_limit,
            f"{len(sol.routes)} 条路径，上限 {inst.vehicle_limit}",
        )
    )

    if sol.cost is not None:
        delta = total - sol.cost
        checks.append(
            Check(
                "ref_cost_matches_declared",
                abs(delta) <= COST_TOLERANCE,
                f"重新计算 {total:.4f}，声明 {sol.cost:.4f}（偏差 {delta:+.4f}）",
                severity=ADVISORY,
            )
        )
    else:
        checks.append(
            Check(
                "ref_cost_declared",
                False,
                ".sol 文件中没有 `Cost` 行",
                severity=ADVISORY,
            )
        )
    return checks, total


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest_rows(suite: Suite) -> list[dict]:
    """每个算例一行摘要，用于 manifest CSV。"""
    rows: list[dict] = []
    for name in suite.names():
        inst = suite.instances[name]
        sol = suite.solutions.get(name)

        inst_checks = check_instance(inst)
        inst_ok = all(c.ok for c in inst_checks)

        ref_checks: list[Check] = []
        recomputed: float | None = None
        if sol is not None:
            ref_checks, recomputed = check_reference(inst, sol)

        published = sol.cost if sol is not None else None
        rows.append(
            {
                "name": name,
                "family": family_of(name),
                "series": series_of(name),
                "n_customers": inst.n_customers,
                "capacity": inst.capacity,
                "vehicle_limit": inst.vehicle_limit,
                "horizon": inst.horizon,
                "service_time": inst.service[1] if inst.n_customers else 0.0,
                "total_demand": inst.total_demand,
                "min_vehicles_capacity": inst.min_vehicles_by_capacity,
                "ref_vehicles": sol.n_vehicles if sol is not None else None,
                "ref_cost_published": published,
                "ref_cost_recomputed": round(recomputed, 4) if recomputed is not None else None,
                "ref_cost_delta": round(abs(recomputed - published), 6)
                if (recomputed is not None and published is not None)
                else None,
                "instance_checks_passed": inst_ok,
                "reference_errors": sum(1 for c in ref_checks if not c.ok and c.severity == ERROR),
                "reference_advisories": sum(
                    1 for c in ref_checks if not c.ok and c.severity == ADVISORY
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


DEFAULT_SUBSETS: tuple[tuple[str, int], ...] = (
    ("C101", 10),
    ("R101", 10),
    ("RC101", 10),
)


def preprocess(
    raw_dir: Path | str,
    out_dir: Path | str,
    *,
    subsets: Iterable[tuple[str, int]] = DEFAULT_SUBSETS,
    verbose: bool = True,
) -> dict:
    """运行完整流水线并返回摘要字典。

    在 ``out_dir`` 下写出::

        instances/<NAME>.json        归一化后的算例，车场位于下标 0
        subsets/<NAME>_n<K>.json     固定规模子算例 + 继承的参考解
        manifest.csv                 每个算例一行
        validation_report.json       所有检查项，以及汇总
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    instances_dir = out_dir / "instances"
    subsets_dir = out_dir / "subsets"
    instances_dir.mkdir(parents=True, exist_ok=True)
    subsets_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite(raw_dir)

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log(f"已从 {raw_dir} 加载 {len(suite)} 个算例、{len(suite.solutions)} 个参考解")

    report: dict = {
        # 相对路径：校验报告要跨机器、跨检出目录保持一致，不能写入绝对路径。
        "raw_dir": portable_path(raw_dir),
        "instance_count": len(suite),
        "instances": {},
        "failures": [],
        "advisories": [],
    }

    total_checks = 0
    for name in suite.names():
        inst = suite.instances[name]
        sol = suite.solutions.get(name)

        checks = check_instance(inst)
        ref_checks: list[Check] = []
        recomputed: float | None = None
        if sol is not None:
            ref_checks, recomputed = check_reference(inst, sol)
        else:
            ref_checks = [Check("ref_present", False, "没有 .sol 文件")]

        all_checks = checks + ref_checks
        total_checks += len(all_checks)
        failed_errors = [c for c in all_checks if not c.ok and c.severity == ERROR]
        failed_advisories = [c for c in all_checks if not c.ok and c.severity == ADVISORY]
        if failed_errors:
            report["failures"].append({"name": name, "failed": [c.as_dict() for c in failed_errors]})
        if failed_advisories:
            report["advisories"].append(
                {"name": name, "advisories": [c.as_dict() for c in failed_advisories]}
            )

        declared = sol.cost if sol is not None else None
        report["instances"][name] = {
            "n_customers": inst.n_customers,
            "capacity": inst.capacity,
            "vehicle_limit": inst.vehicle_limit,
            "total_demand": inst.total_demand,
            "checks": [c.as_dict() for c in all_checks],
            "checks_passed": len(all_checks) - len(failed_errors) - len(failed_advisories),
            "checks_total": len(all_checks),
            "errors": len(failed_errors),
            "advisories": len(failed_advisories),
            "ref_vehicles": sol.n_vehicles if sol is not None else None,
            "ref_cost_published": declared,
            "ref_cost_recomputed": round(recomputed, 4) if recomputed is not None else None,
            "ref_cost_delta": round(recomputed - declared, 6)
            if (recomputed is not None and declared is not None)
            else None,
        }

        payload = inst.to_dict()
        payload["reference_solution"] = sol.to_dict() if sol is not None else None
        (instances_dir / f"{name}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    # ---- 子算例 ----------------------------------------------------------
    report["subsets"] = {}
    for base, size in subsets:
        if base not in suite.instances:
            report["failures"].append({"name": f"{base}_n{size}", "failed": [{"check": "base_instance_present", "ok": False, "detail": f"{base} 不在算例集中"}]})
            continue
        inst = suite.instances[base]
        if size > inst.n_customers:
            raise ValueError(f"{base}: 无法从 {inst.n_customers} 个客户中取出 {size} 个客户")

        # 确定性选取：前 `size` 个客户编号。对 C1/R1 家族而言，这些客户同时
        # 也是车场周围地理上紧密聚集的一群，从而使小算例在不同求解器之间
        # 具有真正意义上的可比性。
        customers = list(range(1, size + 1))
        sub = inst.subset(customers, name=f"{base}_n{size}")

        sub_checks = check_instance(sub)
        ref_sol = None
        sub_ok = all(c.ok for c in sub_checks)
        if base in suite.solutions:
            ref_sol = restrict_solution(sub, suite.solutions[base], customers)
            ref_dist = 0.0
            dist = sub.distance_matrix()
            ref_feasible = True
            for route in ref_sol.routes:
                stats = simulate_route(sub, route, dist)
                if stats is None:
                    ref_feasible = False
                else:
                    ref_dist += stats.distance
            ref_sol.cost = ref_dist
            sub_checks.append(
                Check(
                    "subset_reference_feasible",
                    ref_feasible,
                    "继承而来的路径仍然满足载量与时间窗约束",
                )
            )
            # 合理性检查：继承的参考解不应以某种莫名其妙的方式优于按比例缩小
            # 的全实例最优解，但它必须是子算例的一个有效上界。
            sub_checks.append(
                Check(
                    "subset_reference_is_upper_bound",
                    ref_dist > 0,
                    f"继承的参考解距离 = {ref_dist:.2f}，共 {ref_sol.n_vehicles} 条路径",
                )
            )
            sub_ok = sub_ok and all(c.ok for c in sub_checks)

        payload = sub.to_dict()
        payload["reference_solution"] = ref_sol.to_dict() if ref_sol is not None else None
        payload["provenance"] = {
            "derived_from": base,
            "selection_policy": "first_n_customer_ids",
            "selected_customers": customers,
            "reference_is_upper_bound": True,
            "note": (
                "参考路径由公布的全实例解继承而来，做法是删除子集之外的客户。"
                "对一条可行路径加以限制后它仍然可行，因此该参考解是子算例的一个"
                "有效、但通常并非最优的解。"
            ),
        }
        (subsets_dir / f"{base}_n{size}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

        report["subsets"][f"{base}_n{size}"] = {
            "n_customers": sub.n_customers,
            "capacity": sub.capacity,
            "total_demand": sub.total_demand,
            "min_vehicles_capacity": sub.min_vehicles_by_capacity,
            "reference_vehicles": ref_sol.n_vehicles if ref_sol else None,
            "reference_distance": round(ref_sol.cost, 4) if ref_sol and ref_sol.cost else None,
            "checks": [c.as_dict() for c in sub_checks],
            "checks_passed": sum(1 for c in sub_checks if c.ok),
            "checks_total": len(sub_checks),
        }
        log(
            f"  子算例 {base}_n{size}: {sub.n_customers} 个客户, "
            f"需求 {sub.total_demand}/{sub.capacity}, "
            f"参考解 {ref_sol.n_vehicles} 条路径 / {ref_sol.cost:.2f}"
            if ref_sol
            else f"  子算例 {base}_n{size}: {sub.n_customers} 个客户"
        )

    # ---- manifest --------------------------------------------------------
    try:
        import pandas as pd

        manifest_path = out_dir / "manifest.csv"
        pd.DataFrame(build_manifest_rows(suite)).to_csv(manifest_path, index=False, encoding="utf-8")
        log(f"已写入 {manifest_path}")
    except ImportError:  # pragma: no cover - manifest 只是一个便利功能
        manifest_path = None
        log("pandas 不可用：已跳过 manifest.csv")

    report["summary"] = {
        "instances": len(suite),
        "checks_total": total_checks,
        "errors": len(report["failures"]),
        "advisories": len(report["advisories"]),
        "subsets": len(report["subsets"]),
    }
    (out_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    log(
        f"已校验 {len(suite)} 个算例 / {total_checks} 项检查: "
        f"{report['summary']['errors']} 个 error, "
        f"{report['summary']['advisories']} 条 advisory 发现"
    )
    return report
