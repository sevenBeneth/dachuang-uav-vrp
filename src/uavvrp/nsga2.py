"""从零实现的教科书式 NSGA-II。

为什么不使用框架
----------------
*基线* 的意义在于每个算子都可被检视。本模块严格按照 Deb, Pratap, Agarwal 和
Meyarivan (2002) 在 "A fast and elitist multiobjective genetic algorithm:
NSGA-II", *IEEE Transactions on Evolutionary Computation* 6(2):182-197 中
描述的算法实现：

1. **快速非支配排序** -- 在 O(MN^2) 次目标比较内把种群划分为若干 Pareto 前沿。
2. **拥挤距离** -- 一个前沿内部，每个解周围邻居的疏密程度，用作次级选择准则。
3. **二元锦标赛选择** -- 先比较秩（rank），再比较拥挤距离。
4. **(mu + lambda) 精英保留** -- 父代与子代共同竞争；前沿被整体接纳，直到种群
   即将超出规模，溢出的那个前沿按拥挤距离降序截断。

重复解消除
----------
Deb 等人的存活选择步骤会接纳一个前沿中所有能放得下的成员，而相等的目标向量
互不支配 -- 因此一个由同一个解的副本构成的前沿会被整体接纳。在小规模离散问题
上这是致命的：种群会在若干代内坍缩到一两个不同的解，变异算子再也没有可用
的多样性。因此本实现在存活选择之后额外执行一步（这是可用的 NSGA-II 代码中的
标准做法）：目标向量若已经出现过，其冗余副本会被替换为变异后的变体并重新
评估（轮数由 ``max_dedup_rounds`` 限制）。若要恢复未经修改的流程，设置
``deduplicate=False``。

本模块对编码方式是通用的。问题对象需要提供 ``sample``、``evaluate``、
``crossover`` 和 ``mutate``；因此这里使用的排列型问题得到的是顺序交叉（OX）与
重排变异，而不是实数编码的 SBX 和多项式变异，后者正是排序类问题的标准选择。
*选择* 机制 -- 也就是让它成为 NSGA-II 的那部分 -- 未做任何改动。

每个随机决策都取自同一个带种子的 ``numpy`` 生成器，因此一次运行可由其种子
逐比特复现。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

__all__ = [
    "Problem",
    "NSGA2Result",
    "NSGA2",
    "fast_non_dominated_sort",
    "crowding_distance",
    "hypervolume_2d",
]


class Problem(Protocol):
    """NSGA-II 对优化问题所需的接口。"""

    n_obj: int
    """目标个数，全部为最小化。"""

    def sample(self, rng: np.random.Generator) -> Any:
        """随机抽取一个个体。"""

    def evaluate(self, individual: Any) -> np.ndarray:
        """返回 ``individual`` 的目标向量（长度为 ``n_obj``）。"""

    def crossover(self, a: Any, b: Any, rng: np.random.Generator) -> tuple[Any, Any]:
        """把两个父代重组为两个子代。"""

    def mutate(self, individual: Any, rng: np.random.Generator) -> Any:
        """返回 ``individual`` 的一个变异副本。"""


# ---------------------------------------------------------------------------
# NSGA-II 核心组件
# ---------------------------------------------------------------------------


def fast_non_dominated_sort(F: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """把 ``F``（N x M，最小化）划分为若干 Pareto 前沿。

    返回 ``(fronts, rank)``，其中 ``fronts[0]`` 是非支配解集，
    ``rank[i]`` 是解 ``i`` 所在的前沿序号。目标向量相等的两个解互不支配，
    二者会落在同一个前沿中，这是标准约定。
    """
    F = np.asarray(F, dtype=float)
    n = F.shape[0]
    rank = np.full(n, -1, dtype=int)
    dominated_by_count = np.zeros(n, dtype=int)
    dominates: list[np.ndarray] = [None] * n  # type: ignore[list-item]

    for p in range(n):
        weakly_better = np.all(F <= F[p], axis=1)  # q 在各目标上都不差于 p
        strictly_better = np.any(F < F[p], axis=1)  # ... 且至少在某处更优
        dominated_by_count[p] = int(np.count_nonzero(weakly_better & strictly_better))

        weakly_worse = np.all(F >= F[p], axis=1)
        strictly_worse = np.any(F > F[p], axis=1)
        dominates[p] = np.flatnonzero(weakly_worse & strictly_worse)

    fronts: list[np.ndarray] = []
    current = np.flatnonzero(dominated_by_count == 0)
    rank[current] = 0
    level = 0
    while current.size:
        fronts.append(current)
        promoted: list[int] = []
        for p in current:
            for q in dominates[p]:
                # 这是 n[q] 的标准就地递减；计数数组在每次调用时都会从头重建。
                dominated_by_count[q] -= 1
                if dominated_by_count[q] == 0:
                    rank[q] = level + 1
                    promoted.append(int(q))
        current = np.array(promoted, dtype=int)
        level += 1
    return fronts, rank


def crowding_distance(F_front: np.ndarray) -> np.ndarray:
    """一个前沿中每个成员的拥挤距离（Deb et al., 2002）。

    每个目标上的边界解都取 ``inf``，因此前沿的极值点总会被保留。
    """
    F_front = np.asarray(F_front, dtype=float)
    n, m = F_front.shape
    distance = np.zeros(n, dtype=float)
    if n == 0:
        return distance
    if n <= 2:
        distance[:] = np.inf
        return distance

    for j in range(m):
        order = np.argsort(F_front[:, j], kind="stable")
        values = F_front[order, j]
        span = values[-1] - values[0]
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        if span <= 0.0:
            # 所有成员在该目标上完全相同：它不携带任何多样性信息，因此不做贡献。
            continue
        for k in range(1, n - 1):
            distance[order[k]] += (values[k + 1] - values[k - 1]) / span
    return distance


def hypervolume_2d(F: np.ndarray, reference: Sequence[float]) -> float:
    """两目标最小化前沿的被支配超体积。

    ``reference`` 是 nadir 点：它在两个目标上都必须严格劣于每一个解。
    只有非支配点会做出贡献，因此输入中混入被支配点也无害。
    """
    F = np.asarray(F, dtype=float)
    if F.size == 0:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    if np.any(F > ref):
        raise ValueError("参考点必须支配每一个解")

    order = np.lexsort((F[:, 1], F[:, 0]))  # 先按 f1，再按 f2
    points = F[order]
    volume = 0.0
    previous_f2 = ref[1]
    best_f2 = np.inf
    for f1, f2 in points:
        if f2 >= best_f2:
            continue  # 被更早的点支配
        best_f2 = f2
        if previous_f2 > f2:
            volume += (ref[0] - f1) * (previous_f2 - f2)
            previous_f2 = f2
    return float(volume)


# ---------------------------------------------------------------------------
# 算法
# ---------------------------------------------------------------------------


@dataclass
class NSGA2Result:
    """一次运行的结果。"""

    population: list[Any]
    objectives: np.ndarray
    pareto_front: np.ndarray
    pareto_population: list[Any]
    history: list[dict] = field(default_factory=list)
    n_evaluations: int = 0

    @property
    def front_size(self) -> int:
        return int(self.pareto_front.shape[0])


class NSGA2:
    """NSGA-II 带精英保留的世代进化算法。"""

    def __init__(
        self,
        problem: Problem,
        *,
        pop_size: int = 100,
        n_gen: int = 200,
        seed: int = 0,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.2,
        hypervolume_reference: Sequence[float] | None = None,
        deduplicate: bool = True,
        max_dedup_rounds: int = 10,
        verbose: bool = False,
    ) -> None:
        if pop_size < 2:
            raise ValueError("pop_size must be at least 2")
        if not 0.0 <= crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0.0 <= mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        self.problem = problem
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.seed = seed
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.hypervolume_reference = hypervolume_reference
        self.deduplicate = deduplicate
        self.max_dedup_rounds = max_dedup_rounds
        self.verbose = verbose

    # -- 内部实现 -----------------------------------------------------------

    def _crowding_for_all(self, fronts: list[np.ndarray], F: np.ndarray) -> np.ndarray:
        crowding = np.zeros(F.shape[0], dtype=float)
        for front in fronts:
            crowding[front] = crowding_distance(F[front])
        return crowding

    def _tournament(
        self,
        rank: np.ndarray,
        crowding: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """二元锦标赛：秩更优者胜出，平局时按拥挤距离打破。"""
        a, b = rng.integers(0, rank.size, size=2)
        if rank[a] != rank[b]:
            return int(a if rank[a] < rank[b] else b)
        if crowding[a] != crowding[b]:
            return int(a if crowding[a] > crowding[b] else b)
        return int(a if rng.random() < 0.5 else b)

    def _make_offspring(
        self,
        population: list[Any],
        rank: np.ndarray,
        crowding: np.ndarray,
        rng: np.random.Generator,
    ) -> list[Any]:
        children: list[Any] = []
        while len(children) < self.pop_size:
            i = self._tournament(rank, crowding, rng)
            j = self._tournament(rank, crowding, rng)
            if rng.random() < self.crossover_prob:
                c1, c2 = self.problem.crossover(population[i], population[j], rng)
            else:
                c1, c2 = population[i], population[j]
            if rng.random() < self.mutation_prob:
                c1 = self.problem.mutate(c1, rng)
            if rng.random() < self.mutation_prob:
                c2 = self.problem.mutate(c2, rng)
            children.append(c1)
            if len(children) < self.pop_size:
                children.append(c2)
        return children

    def _deduplicate(
        self,
        population: list[Any],
        F: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        """把已经出现过的目标向量的副本变异掉。

        为每个目标向量保留第一个实现它的个体，其余个体通过变异重新抽取。
        返回额外消耗的评估次数。
        """
        extra = 0
        for _ in range(self.max_dedup_rounds):
            seen: set[tuple] = set()
            redundant: list[int] = []
            for i in range(F.shape[0]):
                key = tuple(np.round(F[i], 9))
                if key in seen:
                    redundant.append(i)
                else:
                    seen.add(key)
            if not redundant:
                break
            for i in redundant:
                population[i] = self.problem.mutate(population[i], rng)
                F[i] = np.asarray(self.problem.evaluate(population[i]), dtype=float)
                extra += 1
        return extra

    def _snapshot(self, generation: int, F: np.ndarray) -> dict:
        fronts, _ = fast_non_dominated_sort(F)
        front = F[fronts[0]]
        # 统计的是不同的目标向量，而不是原始成员数：重复解会一起留在前沿上，
        # 因此原始规模会严重高估前沿的大小。
        distinct = int(np.unique(np.round(front, 9), axis=0).shape[0])
        record = {
            "generation": generation,
            "front_size": distinct,
            "front_members": int(front.shape[0]),
            "distinct_solutions": int(np.unique(np.round(F, 9), axis=0).shape[0]),
            "best_objectives": front.min(axis=0).tolist(),
        }
        if self.hypervolume_reference is not None and F.shape[1] == 2:
            record["hypervolume"] = hypervolume_2d(front, self.hypervolume_reference)
        return record

    # -- 公开 API ----------------------------------------------------------

    def run(self) -> NSGA2Result:
        rng = np.random.default_rng(self.seed)
        problem = self.problem

        population = [problem.sample(rng) for _ in range(self.pop_size)]
        F = np.asarray([problem.evaluate(ind) for ind in population], dtype=float)
        n_evaluations = len(population)

        history: list[dict] = [self._snapshot(0, F)]

        for generation in range(1, self.n_gen + 1):
            fronts, rank = fast_non_dominated_sort(F)
            crowding = self._crowding_for_all(fronts, F)

            offspring = self._make_offspring(population, rank, crowding, rng)
            offspring_F = np.asarray(
                [problem.evaluate(ind) for ind in offspring], dtype=float
            )
            n_evaluations += len(offspring)

            # (mu + lambda)：父代与子代共同竞争。
            combined = population + offspring
            combined_F = np.vstack([F, offspring_F])
            fronts, _ = fast_non_dominated_sort(combined_F)
            crowding = self._crowding_for_all(fronts, combined_F)

            selected: list[int] = []
            for front in fronts:
                if len(selected) + front.size <= self.pop_size:
                    selected.extend(int(i) for i in front)
                    continue
                remaining = self.pop_size - len(selected)
                # 按拥挤距离降序截断溢出的前沿；平局时回退到索引顺序，以保持运行确定。
                order = np.lexsort((front, -crowding[front]))
                selected.extend(int(i) for i in front[order[:remaining]])
                break

            population = [combined[i] for i in selected]
            F = combined_F[selected]
            if self.deduplicate:
                n_evaluations += self._deduplicate(population, F, rng)

            history.append(self._snapshot(generation, F))
            if self.verbose and (generation % 25 == 0 or generation == self.n_gen):
                record = history[-1]
                print(
                    f"  gen {generation:>4}: front={record['front_size']:>3} "
                    f"best={['%.3f' % v for v in record['best_objectives']]}"
                    + (
                        f" hv={record['hypervolume']:.4g}"
                        if "hypervolume" in record
                        else ""
                    ),
                    flush=True,
                )

        fronts, _ = fast_non_dominated_sort(F)
        pareto_idx = fronts[0] if fronts else np.array([], dtype=int)
        # 为便于阅读，把上报的前沿按第一个目标排序。
        if pareto_idx.size:
            pareto_idx = pareto_idx[np.argsort(F[pareto_idx, 0], kind="stable")]
        return NSGA2Result(
            population=population,
            objectives=F,
            pareto_front=F[pareto_idx],
            pareto_population=[population[i] for i in pareto_idx],
            history=history,
            n_evaluations=n_evaluations,
        )
