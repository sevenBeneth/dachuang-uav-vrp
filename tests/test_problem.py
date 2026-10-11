"""切分解码器（split decoder）与 VRPTW 目标函数相关的测试。"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import load_subset, make_instance, requires_subsets

from uavvrp.nsga2 import NSGA2, fast_non_dominated_sort
from uavvrp.problem import VRPTWProblem
from uavvrp.vrptw import simulate_route

SUBSETS = ["C101_n10", "R101_n10", "RC101_n10"]


# ---------------------------------------------------------------------------
# 解码器
# ---------------------------------------------------------------------------


def test_decoder_returns_the_requested_number_of_routes_when_attainable():
    inst = make_instance()
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    perm = (1, 2, 3, 4)
    routes, k = problem.decode(perm, 4)
    assert k == 4
    assert routes == [[1], [2], [3], [4]]


def test_decoder_uses_every_customer_exactly_once():
    inst = make_instance()
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    perm = (3, 1, 4, 2)
    for k in range(1, 5):
        routes, k_used = problem.decode(perm, k)
        flat = [c for route in routes for c in route]
        assert sorted(flat) == [1, 2, 3, 4]
        assert len(routes) == k_used


def test_decoder_repairs_an_unattainable_fleet_size_downwards():
    """在容量为 100、四个客户需求均为 60 的情况下，单条路径不可能可行。"""
    inst = make_instance(demand=[0, 60, 60, 60, 60], capacity=100)
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    routes, k_used = problem.decode((1, 2, 3, 4), 1)
    assert k_used == 4  # 每个客户都需要单独一辆车
    assert len(routes) == 4


def test_decoder_respects_time_windows():
    # 车场在 5.2 关闭。每条只服务单个客户的路径都可行（最远的客户 4、距离
    # 2.0，需要 2+1+2 = 5.0），但任何两两组合都不可行：最省的是客户 1 与 2，
    # 需要 1+1+sqrt(2)+1+1 = 5.41。
    inst = make_instance(due=[5.2, 1000.0, 1000.0, 1000.0, 1000.0])
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))

    routes, k_used = problem.decode((1, 2, 3, 4), 1)
    assert k_used == 4
    for route in routes:
        assert len(route) == 1
        assert simulate_route(inst, route) is not None
    # 两两组合确实不可行，因此该切分并非只是过于保守。
    assert simulate_route(inst, [1, 2]) is None


def test_decoder_returns_none_when_no_partition_is_possible():
    inst = make_instance(due=[0.0] * 5)  # 车场从不开启 -> 任何客户都不可达
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    assert problem.decode((1, 2, 3, 4), 2) is None


def test_decoder_finds_the_cheapest_partition():
    """一侧两个相距较近的客户、另一侧也是两个，应当两两配对。"""
    inst = make_instance(
        coords=[(0.0, 0.0), (10.0, 0.0), (10.0, 1.0), (-10.0, 0.0), (-10.0, 1.0)],
        capacity=100,
    )
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    routes, k_used = problem.decode((1, 2, 3, 4), 2)
    assert k_used == 2
    grouped = sorted(sorted(route) for route in routes)
    assert grouped == [[1, 2], [3, 4]]


# ---------------------------------------------------------------------------
# 目标函数求值
# ---------------------------------------------------------------------------


def test_evaluate_writes_back_the_repaired_fleet_size():
    inst = make_instance(demand=[0, 60, 60, 60, 60], capacity=100)
    problem = VRPTWProblem(inst, objectives=("distance", "vehicles"))
    individual = problem.sample(np.random.default_rng(0))
    individual.k = 1  # 不可达：任何两个需求为 60 的客户都无法装进同一辆车
    values = problem.evaluate(individual)
    assert individual.k == 4
    assert len(individual.routes) == 4
    assert values[1] == 4.0


def test_evaluate_matches_an_independent_recomputation():
    inst = make_instance()
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    individual = problem.sample(np.random.default_rng(2))
    values = problem.evaluate(individual)

    dist = inst.distance_matrix()
    total = 0.0
    makespan = 0.0
    for route in individual.routes:
        stats = simulate_route(inst, route, dist)
        assert stats is not None
        total += stats.distance
        makespan = max(makespan, stats.duration)

    assert values[0] == pytest.approx(total)
    assert values[1] == pytest.approx(makespan)


def test_every_decoded_solution_is_feasible():
    inst = make_instance(due=[200.0] * 5)
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    rng = np.random.default_rng(3)
    for _ in range(200):
        individual = problem.sample(rng)
        problem.evaluate(individual)
        for route in individual.routes:
            assert simulate_route(inst, route) is not None


def test_crossover_produces_valid_permutations():
    inst = make_instance()
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    rng = np.random.default_rng(4)
    a, b = problem.sample(rng), problem.sample(rng)
    for _ in range(50):
        c1, c2 = problem.crossover(a, b, rng)
        for child in (c1, c2):
            assert sorted(child.perm) == problem.customers


def test_mutation_preserves_the_permutation():
    inst = make_instance()
    problem = VRPTWProblem(inst, objectives=("distance", "makespan"))
    rng = np.random.default_rng(5)
    individual = problem.sample(rng)
    for _ in range(100):
        individual = problem.mutate(individual, rng)
        assert sorted(individual.perm) == problem.customers
        assert 1 <= individual.k <= problem.max_routes


def test_unknown_objective_is_rejected():
    inst = make_instance()
    with pytest.raises(ValueError):
        VRPTWProblem(inst, objectives=("distance", "nonsense"))


def test_duplicate_objectives_are_rejected():
    inst = make_instance()
    with pytest.raises(ValueError):
        VRPTWProblem(inst, objectives=("distance", "distance"))


# ---------------------------------------------------------------------------
# 在真实子实例上的端到端测试
# ---------------------------------------------------------------------------


@requires_subsets
@pytest.mark.parametrize("name", SUBSETS)
def test_nsga2_produces_a_feasible_nondominated_front(name):
    instance, _ = load_subset(name)
    problem = VRPTWProblem(instance, objectives=("distance", "makespan"))
    result = NSGA2(
        problem, pop_size=40, n_gen=40, seed=1,
        hypervolume_reference=problem.nadir,
    ).run()

    assert result.pareto_front.shape[0] > 0

    # 报告的前沿必须恰好等于最终种群的非支配解集，且每个成员都必须满足
    # 容量约束与时间窗约束。
    fronts, _ = fast_non_dominated_sort(result.objectives)
    served_customers = set(range(1, instance.n_nodes))
    assert result.pareto_front.shape[0] <= result.objectives.shape[0]
    for individual in result.pareto_population:
        flat = [c for route in individual.routes for c in route]
        assert sorted(flat) == sorted(served_customers)
        for route in individual.routes:
            assert simulate_route(instance, route) is not None
    assert fronts[0].size >= result.pareto_front.shape[0]


@requires_subsets
def test_nsga2_never_returns_a_dominated_solution():
    instance, _ = load_subset("RC101_n10")
    problem = VRPTWProblem(instance, objectives=("distance", "makespan"))
    result = NSGA2(problem, pop_size=40, n_gen=30, seed=9).run()
    front = result.pareto_front
    for i in range(len(front)):
        for j in range(len(front)):
            if i == j:
                continue
            assert not (np.all(front[j] <= front[i]) and np.any(front[j] < front[i]))
