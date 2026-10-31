"""ALNS 参数、重复运行及破坏修复的回归检查。"""

import numpy as np
import pytest

from conftest import make_instance
from uavvrp.drone import DroneProblem, Scenario
from uavvrp.improved import ALNSNSGA2
from uavvrp.problem import VRPTWSolution


def make_problem():
    return DroneProblem(make_instance(), Scenario(
        river_y=.5, river_width=.2, bridge_x=(2,), capacity=20,
        max_distance=100, n_drones=4, max_duration=1000))


@pytest.mark.parametrize('name', ['elite_limit', 'inner_iterations', 'stagnation_limit'])
@pytest.mark.parametrize('value', [0, -1, 1.5, float('nan'), float('inf'), True, '5', None])
def test_alns_budgets_must_be_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        ALNSNSGA2(make_problem(), **{name: value})


def test_numpy_integer_budgets_are_supported():
    engine = ALNSNSGA2(make_problem(), elite_limit=np.int64(2),
                      inner_iterations=np.int64(3), stagnation_limit=np.int64(2))
    assert engine.inner_iterations == 3


@pytest.mark.parametrize('use_schedule', [True, False])
def test_reusing_solver_restores_probabilities_and_complete_history(use_schedule):
    engine = ALNSNSGA2(make_problem(), pop_size=10, n_gen=3, seed=7,
                      crossover_prob=.73, mutation_prob=.23,
                      use_schedule=use_schedule, deduplicate=False)
    first = engine.run()
    second = engine.run()
    assert first.history == second.history
    assert np.array_equal(first.objectives, second.objectives)
    assert first.n_evaluations == second.n_evaluations
    assert second.history[0]['crossover_probability'] == .73
    assert second.history[0]['mutation_probability'] == .23


@pytest.mark.parametrize('destroy', [0, 1])
@pytest.mark.parametrize('repair', [0, 1])
def test_destroy_repair_preserves_customer_coverage_feasibility_and_parent(destroy, repair):
    problem = make_problem()
    engine = ALNSNSGA2(problem)
    for seed in range(20):
        rng = np.random.default_rng(seed)
        parent = problem.sample(rng)
        before_F = problem.evaluate(parent).copy()
        before_routes = [r[:] for r in parent.routes]
        partial, removed = engine._destroy(parent, destroy, rng)
        assert sorted([c for r in partial for c in r] + removed) == problem.customers
        candidate = engine._repair(partial, removed, repair, rng)
        assert candidate is not None
        assert sorted(candidate.perm) == problem.customers
        assert np.isfinite(problem.evaluate(candidate)).all()
        assert candidate.decoded
        assert len(candidate.routes) <= problem.max_routes
        assert all(problem.route_stats(tuple(r)) is not None for r in candidate.routes)
        assert parent.routes == before_routes
        assert np.array_equal(problem.evaluate(parent), before_F)


@pytest.mark.parametrize('trigger_all', [False, True])
def test_all_infeasible_population_skips_alns(trigger_all):
    problem = make_problem()
    engine = ALNSNSGA2(problem, trigger_all=trigger_all)
    population = [VRPTWSolution((1, 2, 3, 4), 2)] * 4
    F = np.full((4, 3), problem.penalty)
    candidates, values, evaluations = engine.enhance(population, F, np.random.default_rng(1), 1)
    assert candidates == []
    assert values.shape == (0, 3)
    assert evaluations == engine.calls == 0
