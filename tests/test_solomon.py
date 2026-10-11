"""解析器、距离约定与校验相关的测试。"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from conftest import RAW_DIR, make_instance, requires_raw

from uavvrp.preprocess import check_instance, check_reference, family_of, restrict_solution, series_of
from uavvrp.solomon import Instance, parse_sol, parse_vrp
from uavvrp.vrptw import EPS, REFERENCE_TOLERANCE, route_distance, simulate_route


# ---------------------------------------------------------------------------
# 命名辅助函数
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,family,series",
    [("C101", "C", "1"), ("C208", "C", "2"), ("R101", "R", "1"),
     ("R211", "R", "2"), ("RC101", "RC", "1"), ("RC208", "RC", "2")],
)
def test_family_and_series(name, family, series):
    assert family_of(name) == family
    assert series_of(name) == series


# ---------------------------------------------------------------------------
# 距离 / 时间约定
# ---------------------------------------------------------------------------


@requires_raw
def test_c101_reference_cost_matches_the_canonical_text():
    """对距离约定的回归锁定（regression lock）。

    广泛流传的经典 C101 解标注为 ``Cost 828.94``。上游文件给出的却是同一组
    路径但包含客户 75，并标注为 827.3（那是另一个更优解的值——见
    docs/data_notes.md）。用不取整的欧氏距离评估所发布的这些路径，必须得到
    828.94（精确到其打印精度）。如果节点编号、车场处理或取整规则有误，结果就
    不可能落在 0.01 的误差范围内。
    """
    inst = parse_vrp(RAW_DIR / "C101.vrp")
    sol = parse_sol(RAW_DIR / "C101.sol")
    dist = inst.distance_matrix()

    total = 0.0
    for route in sol.routes:
        stats = simulate_route(inst, route, dist)
        assert stats is not None, f"路径 {route} 意外地不可行"
        total += stats.distance

    assert total == pytest.approx(828.94, abs=0.01)
    assert len(sol.routes) == 10
    assert sol.n_customers == 100


@requires_raw
def test_depot_has_no_service_time():
    """Solomon 的全局 SERVICE_TIME 作用于客户，而不作用于车场。"""
    inst = parse_vrp(RAW_DIR / "C101.vrp")
    assert inst.service[0] == 0.0
    assert inst.service[1] == 90.0  # 声明的全局客户服务时间


@requires_raw
def test_sol_parser_keeps_route_hash_lines():
    """`#` 是 `Route #N:` 的一部分，不得被当作注释处理。"""
    sol = parse_sol(RAW_DIR / "C101.sol")
    assert len(sol.routes) > 0
    assert sol.cost is not None
    assert all(len(route) > 0 for route in sol.routes)


@requires_raw
def test_every_reference_solution_is_parsed_and_covers_all_customers():
    for vrp in sorted(RAW_DIR.glob("*.vrp")):
        inst = parse_vrp(vrp)
        sol = parse_sol(vrp.with_suffix(".sol"))
        served = [c for route in sol.routes for c in route]
        assert sorted(served) == list(range(1, inst.n_nodes)), inst.name
        assert sol.n_vehicles <= inst.vehicle_limit


@requires_raw
def test_all_instances_pass_the_hard_checks():
    """56 个实例中都不允许存在结构性错误。"""
    failures = {}
    for vrp in sorted(RAW_DIR.glob("*.vrp")):
        inst = parse_vrp(vrp)
        failed = [c for c in check_instance(inst) if not c.ok]
        if failed:
            failures[inst.name] = [c.name for c in failed]
    assert failures == {}


@requires_raw
def test_reference_capacity_is_always_respected():
    """时间窗是上游数据中唯一的缺陷；容量约束绝不允许被破坏。"""
    offenders = []
    for vrp in sorted(RAW_DIR.glob("*.vrp")):
        inst = parse_vrp(vrp)
        sol = parse_sol(vrp.with_suffix(".sol"))
        checks, _ = check_reference(inst, sol)
        capacity = next(c for c in checks if c.name == "ref_routes_feasible_capacity")
        if not capacity.ok:
            offenders.append((inst.name, capacity.detail))
    assert offenders == []


# ---------------------------------------------------------------------------
# 路径仿真
# ---------------------------------------------------------------------------


def test_route_distance_is_symmetric_for_a_two_stop_route():
    inst = make_instance()
    # 车场 -> 1 -> 2 -> 车场 = 1 + sqrt(2) + 1
    expected = 1.0 + math.sqrt(2.0) + 1.0
    assert route_distance(inst, [1, 2]) == pytest.approx(expected)


def test_simulate_route_reports_waiting_and_duration():
    inst = make_instance(
        ready=[0.0, 50.0, 0.0, 0.0, 0.0],
        due=[1000.0] * 5,
        service=[0.0, 5.0, 5.0, 5.0, 5.0],
    )
    stats = simulate_route(inst, [1, 2])
    assert stats is not None
    # 前往客户 1 需耗时 1.0，但其时间窗在 50 才开启 -> 等待 49。
    assert stats.waiting == pytest.approx(49.0)
    # 50（开始服务）+ 5（服务）+ sqrt(2)（前往客户 2）+ 5 + 1（返回车场）
    assert stats.duration == pytest.approx(50.0 + 5.0 + math.sqrt(2.0) + 5.0 + 1.0)


def test_time_window_violation_is_infeasible():
    inst = make_instance(ready=[0.0] * 5, due=[0.0, 0.5, 1000.0, 1000.0, 1000.0])
    assert simulate_route(inst, [1]) is None


def test_capacity_violation_is_infeasible():
    inst = make_instance(demand=[0, 60, 60, 10, 10], capacity=100)
    assert simulate_route(inst, [1, 2]) is None
    assert simulate_route(inst, [1]) is not None


def test_late_return_to_depot_is_infeasible():
    # 车场在 3.0 关闭。服务客户 1 后返回恰好需要
    # 1.0 去程 + 1.0 服务 + 1.0 返程 = 3.0，这是允许的；再增加第二个
    # 客户就会使返回时间超过时间跨度。
    inst = make_instance(due=[3.0, 1000.0, 1000.0, 1000.0, 1000.0])
    assert simulate_route(inst, [1]) is not None
    assert simulate_route(inst, [1, 2]) is None


def test_tolerance_bands_are_separated():
    """求解器使用的精确容差必须远严于数据容差。"""
    assert EPS < 1e-6 < REFERENCE_TOLERANCE


def test_tolerance_parameter_admits_a_marginal_violation():
    inst = make_instance(ready=[0.0] * 5, due=[1000.0, 0.99, 1000.0, 1000.0, 1000.0])
    assert simulate_route(inst, [1]) is None
    assert simulate_route(inst, [1], tolerance=0.5) is not None


# ---------------------------------------------------------------------------
# 子实例
# ---------------------------------------------------------------------------


def test_subset_reindexes_the_depot_to_zero():
    inst = make_instance()
    sub = inst.subset([3, 1], name="SUB")
    assert sub.name == "SUB"
    assert sub.n_customers == 2
    assert sub.coords[0] == inst.coords[0]  # 车场位于首位
    assert sub.coords[1] == inst.coords[3]
    assert sub.coords[2] == inst.coords[1]


def test_subset_rejects_duplicates_and_out_of_range():
    inst = make_instance()
    with pytest.raises(ValueError):
        inst.subset([1, 1])
    with pytest.raises(ValueError):
        inst.subset([99])


def test_restricting_a_solution_keeps_it_feasible():
    inst = make_instance(due=[1000.0] * 5)
    from uavvrp.solomon import Solution

    sol = Solution(name="x", routes=[[1, 2], [3, 4]])
    restricted = restrict_solution(inst, sol, [2, 3])
    assert restricted.routes == [[2], [3]]


# ---------------------------------------------------------------------------
# 经 JSON 的往返转换
# ---------------------------------------------------------------------------


def test_instance_survives_a_json_round_trip():
    inst = make_instance()
    restored = Instance.from_dict(inst.to_dict())
    assert restored == inst


# ---------------------------------------------------------------------------
# 序列化输出的可移植性
# ---------------------------------------------------------------------------


def test_portable_path_is_relative_to_the_project_root():
    """序列化出的源文件路径必须是相对的。

    否则换一台机器、换一个检出目录重新跑预处理，``data/processed`` 下所有 JSON
    都会因为绝对路径不同而产生一整片无意义的 diff。
    """
    from uavvrp.solomon import portable_path, project_root

    root = project_root()
    assert root is not None, "应当能定位到含 pyproject.toml 的项目根目录"
    inside = Path(root) / "data" / "raw" / "solomon" / "C101.vrp"
    assert portable_path(inside) == "data/raw/solomon/C101.vrp"
    # 项目之外的路径退化为文件名，且永远是相对形式。
    assert portable_path(r"C:\elsewhere\deep\C101.vrp") == "C101.vrp"
    assert portable_path("/tmp/elsewhere/R101.vrp") == "R101.vrp"
    assert portable_path("") == ""


def test_to_dict_never_emits_an_absolute_source_path():
    import json as _json

    inst = make_instance()
    payload = _json.dumps(inst.to_dict())
    assert ":\\\\" not in payload and ":/" not in payload.replace("https://", "")
    assert inst.to_dict()["source_path"] == ""
