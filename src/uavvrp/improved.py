"""混合初始化、线性参数调度与精英 ALNS。"""

from math import exp
from numbers import Integral

import numpy as np

from .nsga2 import NSGA2, fast_non_dominated_sort
from .problem import VRPTWSolution


class ALNSNSGA2(NSGA2):
    def __init__(self, *args, use_greedy=True, use_schedule=True, use_alns=True,
                 elite_limit=5, inner_iterations=15, stagnation_limit=5, trigger_all=False, **kwargs):
        super().__init__(*args, **kwargs)
        for name, value in (("elite_limit", elite_limit),
                            ("inner_iterations", inner_iterations),
                            ("stagnation_limit", stagnation_limit)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self._initial_probabilities = (self.crossover_prob, self.mutation_prob)
        self.use_greedy, self.use_schedule, self.use_alns = use_greedy, use_schedule, use_alns
        self.elite_limit, self.inner_iterations, self.stagnation_limit = elite_limit, inner_iterations, stagnation_limit
        self.trigger_all = trigger_all
        self.weights = np.ones((2, 2))
        self.scores = np.zeros((2, 2))
        self.usages = np.zeros((2, 2), dtype=int)
        self.calls = 0
        self.accepted = 0
        self._generation = 0

    def initialize(self, rng):
        self.crossover_prob, self.mutation_prob = self._initial_probabilities
        self.weights[:] = 1
        self.scores[:] = 0
        self.usages[:] = 0
        self.calls = self.accepted = self._generation = 0
        population = super().initialize(rng)
        if self.use_greedy:
            count = self.pop_size//5
            population[:count] = [self.problem.greedy('nearest', rng) for _ in range(count)]
            population[count:2*count] = [self.problem.greedy('deadline', rng) for _ in range(count)]
        return population

    def _make_offspring(self, population, rank, crowding, rng):
        self._generation += 1
        if self.use_schedule:
            progress = self._generation/max(1, self.n_gen)
            self.crossover_prob = .9-.4*progress
            self.mutation_prob = .1*(1-.5*progress)
        return super()._make_offspring(population, rank, crowding, rng)

    def _destroy(self, solution, operator, rng):
        routes = [r[:] for r in solution.routes]
        count = max(1, int(np.ceil(len(solution.perm)*rng.uniform(.05, .1))))
        if operator == 0:
            removed = [int(c) for c in rng.choice(solution.perm, count, replace=False)]
        else:
            contributions = []
            d = self.problem.dist
            for route in routes:
                padded = [0]+route+[0]
                for a, c, b in zip(padded, padded[1:], padded[2:]):
                    contributions.append((d[a][c]+d[c][b]-d[a][b], c))
            removed = [c for _, c in sorted(contributions, reverse=True)[:count]]
        removed_set = set(removed)
        return [[c for c in r if c not in removed_set] for r in routes], removed

    def _repair(self, routes, removed, operator, rng):
        problem = self.problem
        for c in removed:
            choices = []
            for ri, route in enumerate(routes):
                for pos in range(len(route)+1):
                    candidate = route[:pos]+[c]+route[pos:]
                    if problem.route_stats(tuple(candidate), with_risk=False) is None:
                        continue
                    a = route[pos-1] if pos else 0
                    b = route[pos] if pos < len(route) else 0
                    choices.append((problem.dist[a][c]+problem.dist[c][b]-problem.dist[a][b], ri, pos))
            if len(routes) < problem.max_routes and problem.route_stats((c,), with_risk=False) is not None:
                choices.append((2*problem.dist[0][c], len(routes), 0))
            if not choices:
                return None
            _, ri, pos = min(choices) if operator == 0 else choices[int(rng.integers(len(choices)))]
            if ri == len(routes):
                routes.append([])
            routes[ri].insert(pos, c)
        routes = [r for r in routes if r]
        return VRPTWSolution(tuple(c for r in routes for c in r), len(routes))

    def enhance(self, population, F, rng, generation):
        if not self.use_alns:
            return super().enhance(population, F, rng, generation)
        fronts, _ = fast_non_dominated_sort(F)
        candidates, values, evaluations, seen = [], [], 0, set()
        # 归一化标量仅用于 SA，环境选择仍用原始多目标支配关系。
        feasible_F = F[np.all(F < self.problem.penalty, axis=1)]
        if not len(feasible_F):
            return [], np.empty((0, self.problem.n_obj)), 0
        scale = np.maximum(np.ptp(feasible_F, axis=0), np.maximum(np.abs(feasible_F).max(axis=0)*.01, 1))
        indices = range(len(population)) if self.trigger_all else fronts[0]
        for idx in indices:
            key = tuple(np.round(F[idx], 9))
            if (not self.trigger_all and key in seen) or not population[idx].decoded:
                continue
            seen.add(key)
            if not self.trigger_all and len(seen) > self.elite_limit:
                break
            current, current_F = population[idx], F[idx].copy()
            temperature, stagnation = .5, 0
            self.calls += 1
            for _ in range(self.inner_iterations):
                operators = [int(rng.choice(2, p=w/w.sum())) for w in self.weights]
                d, r = operators
                self.usages[0, d] += 1
                self.usages[1, r] += 1
                partial, removed = self._destroy(current, d, rng)
                candidate = self._repair(partial, removed, r, rng)
                reward = 0.0
                stagnation += 1
                if candidate is not None:
                    new_F = self.problem.evaluate(candidate)
                    evaluations += 1
                    if candidate.decoded:
                        # 保留沿途候选，SA 接受劣解也不会覆盖已有精英。
                        candidates.append(candidate)
                        values.append(new_F)
                        delta = float(np.sum((new_F-current_F)/scale))
                        if delta <= 0 or rng.random() < exp(-delta/temperature):
                            dominance = np.all(new_F <= current_F) and np.any(new_F < current_F)
                            reward = 10 if dominance else (5 if delta < 0 else 1)
                            if delta < -1e-12:
                                stagnation = 0
                            current, current_F = candidate, new_F
                            self.accepted += 1
                for row, op in enumerate(operators):
                    self.scores[row, op] += reward
                    average = self.scores[row, op]/self.usages[row, op]
                    self.weights[row, op] = max(1e-6, .9*self.weights[row, op]+.1*average)
                temperature *= .95
                if stagnation >= self.stagnation_limit:
                    break
        return candidates, np.asarray(values).reshape(-1, self.problem.n_obj), evaluations

    def _snapshot(self, generation, F):
        record = super()._snapshot(generation, F)
        record.update(alns_calls=self.calls, alns_accepted=self.accepted,
                      operator_usages=self.usages.tolist(), operator_weights=self.weights.tolist(),
                      crossover_probability=self.crossover_prob, mutation_probability=self.mutation_prob)
        return record
