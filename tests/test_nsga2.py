"""NSGA-II 选择机制相关的测试。"""

from __future__ import annotations

import numpy as np
import pytest

from uavvrp.nsga2 import (
    NSGA2,
    crowding_distance,
    fast_non_dominated_sort,
    hypervolume_2d,
)


# ---------------------------------------------------------------------------
# 快速非支配排序
# ---------------------------------------------------------------------------


def test_non_dominated_sort_separates_known_fronts():
    F = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ]
    )
    fronts, rank = fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1]
    assert fronts[1].tolist() == [2]
    assert fronts[2].tolist() == [3]
    assert rank.tolist() == [0, 0, 1, 2]


def test_identical_solutions_share_a_front():
    """目标向量相等的解互不支配，因此二者都留在第 0 层前沿。"""
    F = np.array([[1.0, 1.0], [1.0, 1.0]])
    fronts, rank = fast_non_dominated_sort(F)
    assert sorted(fronts[0].tolist()) == [0, 1]
    assert rank.tolist() == [0, 0]


def test_every_solution_lands_in_exactly_one_front():
    rng = np.random.default_rng(7)
    F = rng.random((60, 3))
    fronts, rank = fast_non_dominated_sort(F)
    placed = sorted(i for front in fronts for i in front)
    assert placed == list(range(60))
    assert (rank >= 0).all()


def test_fronts_are_mutually_non_dominating_and_ordered():
    rng = np.random.default_rng(11)
    F = rng.random((80, 2))
    fronts, rank = fast_non_dominated_sort(F)
    # 较前层前沿中的任一成员，必须支配较后层前沿的每一个成员，或与其不可比较；
    # 较后的解永远不能支配较前的解。
    for earlier, later in zip(fronts, fronts[1:]):
        for p in earlier:
            for q in later:
                assert not (np.all(F[q] <= F[p]) and np.any(F[q] < F[p]))


# ---------------------------------------------------------------------------
# 拥挤度距离
# ---------------------------------------------------------------------------


def test_crowding_distance_marks_the_boundaries_infinite():
    F = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
    distance = crowding_distance(F)
    assert np.isinf(distance[0]) and np.isinf(distance[3])
    assert np.isfinite(distance[1]) and np.isfinite(distance[2])


def test_crowding_distance_prefers_isolated_solutions():
    """邻居相距较远的解，比密集聚簇中的解拥挤度更低。"""
    F = np.array([[0.0, 10.0], [5.0, 5.0], [5.2, 4.9], [10.0, 0.0]])
    distance = crowding_distance(F)
    assert distance[1] > 0 and distance[2] > 0
    assert distance[0] == np.inf and distance[3] == np.inf


def test_crowding_distance_of_tiny_fronts_is_infinite():
    assert np.isinf(crowding_distance(np.array([[1.0, 1.0]]))).all()
    assert np.isinf(crowding_distance(np.array([[1.0, 1.0], [2.0, 2.0]]))).all()


def test_crowding_distance_handles_a_degenerate_objective():
    """恒定的目标列不得因跨度为 0 而产生 NaN。"""
    F = np.array([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]])
    distance = crowding_distance(F)
    assert not np.isnan(distance).any()


# ---------------------------------------------------------------------------
# Hypervolume
# ---------------------------------------------------------------------------


def test_hypervolume_of_a_single_point():
    assert hypervolume_2d(np.array([[1.0, 1.0]]), (2.0, 2.0)) == pytest.approx(1.0)


def test_hypervolume_of_a_two_point_front_with_overlap():
    # [0,3]x[2,3] 与 [2,3]x[0,3] 的并集 = 3 + 3 - 1 = 5。
    assert hypervolume_2d(np.array([[0.0, 2.0], [2.0, 0.0]]), (3.0, 3.0)) == pytest.approx(5.0)


def test_hypervolume_ignores_dominated_points():
    dominated = np.array([[0.0, 2.0], [2.0, 0.0], [2.5, 2.5]])
    assert hypervolume_2d(dominated, (3.0, 3.0)) == pytest.approx(5.0)


def test_hypervolume_is_monotone_under_a_better_front():
    reference = (10.0, 10.0)
    worse = np.array([[2.0, 2.0]])
    better = np.array([[1.0, 1.0]])
    assert hypervolume_2d(better, reference) > hypervolume_2d(worse, reference)


def test_hypervolume_rejects_a_bad_reference_point():
    with pytest.raises(ValueError):
        hypervolume_2d(np.array([[5.0, 5.0]]), (1.0, 1.0))


# ---------------------------------------------------------------------------
# 在玩具问题（toy problem）上运行算法
# ---------------------------------------------------------------------------


class _ToyProblem:
    """在 x 属于 [-4, 4] 上最小化 (x^2, (x-2)^2)。真实前沿为 x 属于 [0, 2]。"""

    n_obj = 2

    def sample(self, rng):
        return float(rng.uniform(-4.0, 4.0))

    def evaluate(self, individual):
        x = individual
        return np.array([x * x, (x - 2.0) ** 2])

    def crossover(self, a, b, rng):
        w = rng.random()
        return w * a + (1 - w) * b, (1 - w) * a + w * b

    def mutate(self, individual, rng):
        return float(np.clip(individual + rng.normal(0.0, 0.2), -4.0, 4.0))


def test_nsga2_approaches_the_true_front_on_a_toy_problem():
    problem = _ToyProblem()
    result = NSGA2(
        problem, pop_size=60, n_gen=150, seed=3,
        crossover_prob=0.9, mutation_prob=0.5,
        hypervolume_reference=(20.0, 20.0),
    ).run()

    assert result.pareto_front.shape[0] >= 5
    f1 = result.pareto_front[:, 0]
    f2 = result.pareto_front[:, 1]
    # 解析前沿是 x 属于 [0, 2]，其中 f1 = x^2、f2 = (2 - x)^2，因此在前沿上
    # sqrt(f1) + sqrt(f2) 恰好等于 2，而在前沿之外则大于 2。（注意仅 f1 的
    # 取值范围就是 [0, 4]：前沿的 x = 2 一端正是 f1 的最大值。）
    assert np.allclose(np.sqrt(f1) + np.sqrt(f2), 2.0, atol=0.05)
    assert f1.min() < 0.01
    assert f2.min() < 0.01
    assert len(result.history) == 151
    # pop_size 个初始个体，加上每一代的 pop_size 个个体，再加上因重复个体
    # 淘汰而额外消耗的评估次数。
    assert result.n_evaluations >= 60 + 150 * 60
    assert all("front_size" in record for record in result.history)


def test_nsga2_is_reproducible_from_its_seed():
    problem = _ToyProblem()
    kwargs = dict(pop_size=30, n_gen=20, seed=42, hypervolume_reference=(20.0, 20.0))
    first = NSGA2(problem, **kwargs).run()
    second = NSGA2(problem, **kwargs).run()
    assert np.array_equal(first.pareto_front, second.pareto_front)


def test_nsga2_different_seeds_give_different_trajectories():
    problem = _ToyProblem()
    a = NSGA2(problem, pop_size=20, n_gen=5, seed=1).run()
    b = NSGA2(problem, pop_size=20, n_gen=5, seed=2).run()
    assert not np.array_equal(a.objectives, b.objectives)


def test_nsga2_population_never_exceeds_the_requested_size():
    problem = _ToyProblem()
    result = NSGA2(problem, pop_size=25, n_gen=10, seed=5).run()
    assert result.objectives.shape[0] == 25


@pytest.mark.parametrize("pop_size", [1, 0])
def test_nsga2_rejects_a_degenerate_population_size(pop_size):
    with pytest.raises(ValueError):
        NSGA2(_ToyProblem(), pop_size=pop_size)
