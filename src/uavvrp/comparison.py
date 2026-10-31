"""为 MOEA/D 与 SPEA2 提供相同的排列编码和配送评价接口。"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.optimize import minimize
from pymoo.util.ref_dirs import get_reference_directions

from .problem import VRPTWSolution


def unpack(x):
    return VRPTWSolution(tuple(int(c) for c in x[:-1]), int(x[-1]))


def pack(solution):
    return np.asarray((*solution.perm, solution.k), dtype=int)


class Adapter(ElementwiseProblem):
    def __init__(self, problem):
        super().__init__(n_var=problem.n_customers+1, n_obj=3, xl=0, xu=problem.n_customers)
        self.drone = problem

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.drone.evaluate(unpack(x))


class PermutationSampling(Sampling):
    def _do(self, problem, n_samples, random_state=None, **kwargs):
        return np.asarray([pack(problem.drone.sample(random_state)) for _ in range(n_samples)])


class OrderCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 2, prob=.9)

    def _do(self, problem, X, random_state=None, **kwargs):
        output = np.empty_like(X)
        for i in range(X.shape[1]):
            a, b = problem.drone.crossover(unpack(X[0, i]), unpack(X[1, i]), random_state)
            output[0, i], output[1, i] = pack(a), pack(b)
        return output


class PermutationMutation(Mutation):
    def __init__(self):
        super().__init__(prob=.2)

    def _do(self, problem, X, random_state=None, **kwargs):
        return np.asarray([pack(problem.drone.mutate(unpack(x), random_state)) for x in X])


class Recorder(Callback):
    def __init__(self):
        super().__init__()
        self.frames = []

    def notify(self, algorithm):
        self.frames.append({'generation': int(algorithm.n_gen),
                            'front': algorithm.opt.get('F').tolist(),
                            'evaluations': int(algorithm.evaluator.n_eval)})


def run_comparator(name, problem, pop_size, generations, seed):
    options = dict(sampling=PermutationSampling(), crossover=OrderCrossover(), mutation=PermutationMutation())
    if name == 'MOEAD':
        directions = get_reference_directions('energy', 3, pop_size, seed=seed)
        algorithm = MOEAD(ref_dirs=directions, n_neighbors=min(15, pop_size), **options)
    elif name == 'SPEA2':
        algorithm = SPEA2(pop_size=pop_size, **options)
    else:
        raise ValueError(name)
    recorder = Recorder()
    # pymoo 第一代为初始化；额外加一代以与本项目 n_gen 次繁殖相同。
    result = minimize(Adapter(problem), algorithm, ('n_gen', generations+1), seed=seed,
                      callback=recorder, verbose=False)
    solutions = [unpack(x) for x in result.pop.get('X')]
    F = np.asarray([problem.evaluate(s) for s in solutions])
    return solutions, F, recorder.frames, int(result.algorithm.evaluator.n_eval)
