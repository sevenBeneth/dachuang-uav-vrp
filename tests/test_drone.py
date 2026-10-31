from dataclasses import replace

import numpy as np
import pytest

from conftest import make_instance
from uavvrp.drone import DroneProblem, RiverNetwork, Scenario, fuzzy_selection
from uavvrp.improved import ALNSNSGA2
from uavvrp.metrics import indicators, paired_test
from uavvrp.nsga2 import NSGA2
from uavvrp.problem import VRPTWProblem, VRPTWSolution


def scenario(**kwargs):
    return replace(Scenario(river_y=.5, river_width=.2, bridge_x=(2,),
                            capacity=100, max_distance=100, n_drones=4,
                            max_duration=1000, risk_period=20, risk_decay=2), **kwargs)


def test_crossing_uses_both_bridge_banks_and_recomputes_distance():
    network = RiverNetwork([(0, 0), (0, 1)], scenario())
    assert network.paths[0][1] == ((0, 0), (2, .4), (2, .6), (0, 1))
    expected = 2*np.hypot(2, .4)+.2
    assert network.dist[0][1] == pytest.approx(expected)
    assert network.dist[1][0] == pytest.approx(expected)
    assert network.dist[0][0] == 0


def test_same_bank_is_direct_and_no_bridge_disconnects_banks():
    network = RiverNetwork([(0, 0), (1, 0), (0, 1)], scenario(bridge_x=()))
    assert network.dist[0][1] == 1
    assert np.isinf(network.dist[0][2])


def test_water_nodes_are_rejected():
    with pytest.raises(ValueError, match='水域'):
        RiverNetwork([(0, .5)], scenario())


@pytest.mark.parametrize('change', [dict(speed=0), dict(max_distance=-1), dict(river_width=0),
                                    dict(risk_period=float('nan')), dict(late_weight=-1)])
def test_invalid_scenario(change):
    with pytest.raises(ValueError):
        scenario(**change)


def test_return_leg_counts_toward_endurance():
    inst = make_instance(coords=[(0, 0), (10, 0)])
    problem = DroneProblem(inst, scenario(max_distance=15))
    solution = VRPTWSolution((1,), 1)
    assert np.all(problem.evaluate(solution) == 1e12)
    assert not solution.decoded


def test_waiting_counts_toward_duration_and_soft_time_cost():
    inst = make_instance(coords=[(0, 0), (1, 0)], ready=[0, 10], due=[100, 20], service=[0, 2])
    problem = DroneProblem(inst, scenario())
    stats = problem.route_stats((1,))
    assert stats['time'] == 10  # 到达 1 + 提前惩罚 9
    assert stats['duration'] == 13  # 飞行 1 + 等待 9 + 服务 2 + 返航 1
    assert stats['distance'] == 2
    assert DroneProblem(inst, scenario(max_duration=12)).route_stats((1,)) is None


def test_urgent_lateness_uses_urgent_weight():
    inst = make_instance(coords=[(0, 0), (10, 0)], due=[100, 2])
    normal = DroneProblem(inst, scenario()).route_stats((1,))
    urgent = DroneProblem(inst, scenario(urgent=(1,))).route_stats((1,))
    assert normal['time'] == 10+2*8
    assert urgent['time'] == 10+5*8


def test_capacity_and_fleet_limit_are_hard_constraints():
    inst = make_instance(demand=[0, 60, 60, 60, 60])
    problem = DroneProblem(inst, scenario(n_drones=3))
    solution = VRPTWSolution((1, 2, 3, 4), 1)
    problem.evaluate(solution)
    assert not solution.decoded


def test_risk_changes_with_departure_time_and_is_nonnegative():
    network = RiverNetwork([(0, 0), (1, 0)], scenario())
    assert network.risk(0, 1, 5) > network.risk(0, 1, 15) >= 0
    xx, yy, risk = network.risk_map(5, 10)
    assert xx.shape == yy.shape == risk.shape == (10, 10)
    assert np.nanmin(risk) >= 0 and np.nanmax(risk) <= 1


def test_precomputed_risk_matches_independent_midpoint_integration():
    network = RiverNetwork([(0, 0), (1, 1)], scenario(obstacles=((0, 0),)))
    departure, expected = 3.0, 0.0
    for a, b in zip(network.paths[0][1], network.paths[0][1][1:]):
        dt = np.hypot(b[0]-a[0], b[1]-a[1])/network.scenario.speed
        for k in range(network.scenario.risk_samples):
            u = (k+.5)/network.scenario.risk_samples
            expected += network.risk_at(a[0]+u*(b[0]-a[0]), a[1]+u*(b[1]-a[1]), departure+u*dt)*dt/network.scenario.risk_samples
        departure += dt
    assert network.risk(0, 1, 3) == pytest.approx(expected)


def test_all_source_reference_costs_and_routes_match_truncated_distances(raw_dir):
    from uavvrp.preprocess import check_reference
    from uavvrp.solomon import parse_sol, parse_vrp
    for path in sorted(raw_dir.glob('*.vrp')):
        inst, sol = parse_vrp(path), parse_sol(path.with_suffix('.sol'))
        checks, _ = check_reference(inst, sol)
        for name in ('ref_cost_matches_declared', 'ref_routes_feasible_source'):
            check = next(c for c in checks if c.name == name)
            assert check.ok, (path.name, check)


def test_risk_includes_return_leg_at_actual_departure_time():
    inst = make_instance(coords=[(0, 0), (1, 0)], service=[0, 2])
    problem = DroneProblem(inst, scenario())
    stats = problem.route_stats((1,))
    expected = problem.network.risk(0, 1, 0)+problem.network.risk(1, 0, 3)
    assert stats['risk'] == pytest.approx(expected)


def test_feasible_solution_metrics_match_route_recomputation():
    problem = DroneProblem(make_instance(), scenario())
    sol = problem.sample(np.random.default_rng(2))
    F = problem.evaluate(sol)
    stats = [problem.route_stats(tuple(r)) for r in sol.routes]
    assert sorted(c for r in sol.routes for c in r) == [1, 2, 3, 4]
    assert F == pytest.approx([sum(s[k] for s in stats) for k in ('time', 'distance', 'risk')])
    assert problem.evaluate(sol) == pytest.approx(F)


def test_alns_is_seed_reproducible_and_retains_elites():
    problem = DroneProblem(make_instance(), scenario())
    engines = [ALNSNSGA2(problem, pop_size=10, n_gen=4, seed=3, deduplicate=False) for _ in range(2)]
    a, b = [engine.run() for engine in engines]
    assert np.array_equal(a.objectives, b.objectives)
    assert engines[0].calls > 0
    assert engines[0].usages.sum() > 0
    assert np.all(engines[0].weights > 0)
    assert a.n_evaluations > 50
    assert engines[0].crossover_prob == pytest.approx(.5)
    assert engines[0].mutation_prob == pytest.approx(.05)
    for s in a.population:
        assert s.decoded
    initial = np.asarray(a.history[0]['front'])
    final = a.pareto_front
    assert all(np.any(np.all(final <= point+1e-9, axis=1)) for point in initial)


def test_ablation_disables_alns_evaluations():
    problem = DroneProblem(make_instance(), scenario())
    engine = ALNSNSGA2(problem, pop_size=10, n_gen=2, use_alns=False, deduplicate=False)
    result = engine.run()
    assert result.n_evaluations == 30
    assert engine.calls == 0


def test_full_trigger_processes_all_feasible_individuals_including_duplicates():
    problem = DroneProblem(make_instance(), scenario())
    population = [problem.greedy('nearest', np.random.default_rng(1)) for _ in range(8)]
    F = np.asarray([problem.evaluate(s) for s in population])
    elite = ALNSNSGA2(problem, pop_size=8, n_gen=1, inner_iterations=1)
    full = ALNSNSGA2(problem, pop_size=8, n_gen=1, inner_iterations=1, trigger_all=True)
    elite.enhance(population, F, np.random.default_rng(2), 1)
    full.enhance(population, F, np.random.default_rng(2), 1)
    assert elite.calls == 1
    assert full.calls == 8


def test_fuzzy_selection_handles_constant_objectives_and_weights():
    F = [[1, 8, 2], [3, 2, 2]]
    index, scores = fuzzy_selection(F, [0, 1, 0])
    assert index == 1
    assert np.isfinite(scores).all()
    with pytest.raises(ValueError):
        fuzzy_selection(F, [-1, 0, 0])


def test_metrics_use_common_scale_and_empirical_reference():
    reference = np.array([[0, 0, 0]])
    a = indicators(reference, reference, np.zeros(3), np.ones(3))
    b = indicators([[.5, .5, .5]], reference, np.zeros(3), np.ones(3))
    assert a['hv'] == pytest.approx(1.1**3)
    assert a['igd'] == 0
    assert a['hv'] > b['hv']
    assert paired_test([1, 1], [1, 1]) == 1


def test_baseline_rejects_missing_or_duplicate_customers_and_handles_singleton():
    problem = VRPTWProblem(make_instance())
    with pytest.raises(ValueError):
        problem.decode((1, 1, 2, 3), 2)
    inst = make_instance(coords=[(0, 0), (1, 0)])
    problem = VRPTWProblem(inst)
    result = NSGA2(problem, pop_size=4, n_gen=2).run()
    assert result.front_size > 0


@pytest.mark.parametrize('algorithm', ['MOEAD', 'SPEA2'])
def test_comparators_preserve_permutations_and_are_reproducible(algorithm):
    from uavvrp.comparison import run_comparator
    problem = DroneProblem(make_instance(), scenario())
    a = run_comparator(algorithm, problem, 6, 2, 5)
    b = run_comparator(algorithm, problem, 6, 2, 5)
    assert np.array_equal(a[1], b[1])
    assert all(sorted(s.perm) == [1, 2, 3, 4] for s in a[0])
