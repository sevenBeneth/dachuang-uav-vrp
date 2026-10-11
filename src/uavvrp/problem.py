"""NSGA-II 所求解的 VRPTW 问题，含编码方式与解码器。

编码
--------------------------------------------------------------------------
一个个体是 ``(perm, k)``：

* ``perm`` -- 所有客户 id 的一个排列，即巨型路径（giant tour）；
* ``k`` -- 请求使用的车辆数。

解码器
--------------------------------------------------------------------------
``perm`` 由一个动态规划恰好切成 ``k`` 段连续路径，该动态规划在满足车辆容量和
时间窗的前提下最小化总行驶距离（即 Prins, 2004 的经典 "split" 过程）。因此每个
解码后的解在构造上就是可行的：没有修复步骤，没有惩罚项，也不可能有个体不可行
却进入 Pareto 前沿。

如果某个 ``perm`` 取不到 ``k``（可行的路径数太少或太多），解码器会把它修复为
**最接近的可行取值**，并记录实际使用的值。基因型会被就地更新，因此基因型与
表型始终保持一致。

目标
--------------------------------------------------------------------------
所有目标均为最小化：

``distance``  所有路径的总行驶距离
``vehicles``  使用的路径条数（经典的车辆数 / 距离权衡）
``makespan``  最长单条路径的持续时间（行驶 + 服务 + 等待）
``waiting``   等待时间窗开启所花的总时间

其中任意两个都可以被选中，因此同一套机制既能产生经典的（车辆数 vs. 距离）
前沿，也能产生（距离 vs. 完工时间）前沿。

范围说明
--------------------------------------------------------------------------
这是论文对比所用的 *标准* 基线。它只建模单一车型，不含跨江限制，也不含无人机
续航限制。项目自身的贡献 -- 每架无人机的载重/续航限制，以及禁止直线飞越江面的
跨江飞行约束 -- 属于改进后的算法，此处刻意不包含，以便第二阶段的改进有一个
干净的参照点。``VRPTWProblem`` 对每个解都暴露 ``routes`` 和 ``metrics``，
这类约束正是在那里施加的。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .solomon import Instance
from .vrptw import EPS, simulate_route

__all__ = ["VRPTWSolution", "VRPTWProblem", "OBJECTIVE_NAMES"]

OBJECTIVE_NAMES = ("distance", "vehicles", "makespan", "waiting")

# 完全无法解码的个体所得到的目标值。它远差于任何真实解，因此这类个体总会被支配。
INFEASIBLE_OBJECTIVE = 1e12


@dataclass
class VRPTWSolution:
    """一个个体：基因型，以及解码器由此导出的一切。"""

    perm: tuple[int, ...]
    k: int
    routes: list[list[int]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    decoded: bool = False

    def copy(self) -> "VRPTWSolution":
        return VRPTWSolution(perm=self.perm, k=self.k)

    @property
    def k_used(self) -> int:
        return len(self.routes)


class VRPTWProblem:
    """基于 :class:`Instance` 的、经 split 解码的多目标 VRPTW。"""

    def __init__(
        self,
        instance: Instance,
        *,
        objectives: Sequence[str] = ("distance", "vehicles"),
        penalty: float = INFEASIBLE_OBJECTIVE,
    ) -> None:
        unknown = [name for name in objectives if name not in OBJECTIVE_NAMES]
        if unknown:
            raise ValueError(
                f"未知的目标 {unknown}；请从 {list(OBJECTIVE_NAMES)} 中选择"
            )
        if len(objectives) < 2:
            raise ValueError("NSGA-II 至少需要两个目标")
        if len(set(objectives)) != len(objectives):
            raise ValueError("各目标之间必须互不相同")

        self.instance = instance
        self.dist = instance.distance_matrix()
        self.objectives = tuple(objectives)
        self.n_obj = len(self.objectives)
        self.penalty = penalty

        self.customers = list(range(1, instance.n_nodes))
        self.n_customers = len(self.customers)
        self.max_routes = max(1, min(self.n_customers, instance.vehicle_limit))

        self.nadir = np.array([self._nadir_value(name) for name in self.objectives])

    # -- 边界 --------------------------------------------------------------

    def _nadir_value(self, objective: str) -> float:
        """由实例导出的廉价上界，用作超体积的 nadir 点。"""
        if objective == "distance":
            # 每个客户单独一辆车：没有任何划分会比这更长。
            return sum(2.0 * self.dist[0][c] for c in self.customers) + 1.0
        if objective == "vehicles":
            return float(self.max_routes)
        if objective == "makespan":
            return float(self.instance.due[0]) + 1.0
        if objective == "waiting":
            return float(self.instance.due[0]) * self.max_routes + 1.0
        raise ValueError(objective)  # pragma: no cover

    # -- 解码器 ------------------------------------------------------------

    def _segment_table(self, perm: Sequence[int]) -> list[list[float]]:
        """``table[i][j]`` = 可行路径 ``perm[i:j]`` 的距离，否则为 inf。

        增量式构建。当累计载重超过容量、或某个到达时间晚于最晚时间时，扫描可以
        提前停止，因为二者对停靠点数都是单调的。但晚回车场 *不* 单调（更远的
        停靠点反而可能离车场更近），因此这种情况只把该表项标记为不可行，扫描
        仍会继续延伸。
        """
        n = len(perm)
        inf = float("inf")
        table = [[inf] * (n + 1) for _ in range(n + 1)]
        inst = self.instance
        dist = self.dist

        for i in range(n):
            load = 0
            clock = inst.ready[0]
            previous = 0
            travelled = 0.0
            for j in range(i + 1, n + 1):
                customer = perm[j - 1]
                leg = dist[previous][customer]
                arrival = clock + leg
                if arrival > inst.due[customer] + EPS:
                    break  # 时间只会增长，因此更长的每一段也都会迟到
                start = arrival if arrival >= inst.ready[customer] else inst.ready[customer]
                clock = start + inst.service[customer]
                load += inst.demand[customer]
                if load > inst.capacity:
                    break  # 载重只会增长
                travelled += leg
                previous = customer
                if clock + dist[customer][0] <= inst.due[0] + EPS:
                    table[i][j] = travelled + dist[customer][0]
        return table

    def decode(self, perm: Sequence[int], k_requested: int) -> tuple[list[list[int]], int] | None:
        """把 ``perm`` 恰好切成 ``k_requested`` 条路径，并优先取最省的方案。

        返回 ``(routes, k_used)``；若完全不存在任何划分，则返回 ``None``。
        """
        n = len(perm)
        kmax = min(self.max_routes, n)
        inf = float("inf")
        table = self._segment_table(perm)

        # dp[j][kk] = 用 kk 条路径服务前 j 个停靠点的最省方案。
        dp = [[inf] * (kmax + 1) for _ in range(n + 1)]
        parent = [[-1] * (kmax + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for kk in range(1, kmax + 1):
            for j in range(1, n + 1):
                best = inf
                best_i = -1
                for i in range(j - 1, -1, -1):
                    base = dp[i][kk - 1]
                    if base == inf:
                        continue
                    segment = table[i][j]
                    if segment == inf:
                        continue
                    value = base + segment
                    if value < best:
                        best = value
                        best_i = i
                dp[j][kk] = best
                parent[j][kk] = best_i

        attainable = [kk for kk in range(1, kmax + 1) if dp[n][kk] < inf]
        if not attainable:
            return None

        # 把请求的车辆数修复为最接近的可行取值。
        kk_used = min(attainable, key=lambda kk: (abs(kk - k_requested), kk))

        routes: list[list[int]] = []
        j, kk = n, kk_used
        while kk > 0:
            i = parent[j][kk]
            routes.append(list(perm[i:j]))
            j, kk = i, kk - 1
        routes.reverse()
        return routes, kk_used

    # -- Problem 协议 ------------------------------------------------------

    def sample(self, rng: np.random.Generator) -> VRPTWSolution:
        perm = rng.permutation(self.customers)
        k = int(rng.integers(1, self.max_routes + 1))
        return VRPTWSolution(perm=tuple(int(c) for c in perm), k=k)

    def evaluate(self, individual: VRPTWSolution) -> np.ndarray:
        decoded = self.decode(individual.perm, individual.k)
        if decoded is None:
            individual.routes = []
            individual.metrics = {}
            individual.decoded = False
            return np.full(self.n_obj, self.penalty, dtype=float)

        routes, k_used = decoded
        individual.routes = routes
        individual.k = k_used  # 拉马克式修复让基因型与表型保持一致
        individual.decoded = True

        total_distance = 0.0
        total_waiting = 0.0
        makespan = 0.0
        for route in routes:
            stats = simulate_route(self.instance, route, self.dist)
            if stats is None:  # pragma: no cover - 解码器保证可行性
                raise AssertionError(
                    f"{self.instance.name}：解码器产生了不可行的路径 {route}"
                )
            total_distance += stats.distance
            total_waiting += stats.waiting
            makespan = max(makespan, stats.duration)

        metrics = {
            "distance": total_distance,
            "vehicles": float(len(routes)),
            "makespan": makespan,
            "waiting": total_waiting,
        }
        individual.metrics = metrics
        return np.array([metrics[name] for name in self.objectives], dtype=float)

    def _order_crossover(
        self, p1: Sequence[int], p2: Sequence[int], rng: np.random.Generator
    ) -> list[int]:
        """排列的顺序交叉（OX）：保留 p1 的一段，其余从 p2 填充。"""
        n = len(p1)
        a, b = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
        child: list[int | None] = [None] * n
        child[a:b] = p1[a:b]
        kept = set(p1[a:b])
        filler = [c for c in p2 if c not in kept]
        fill_positions = list(range(b, n)) + list(range(0, a))
        for position, value in zip(fill_positions, filler):
            child[position] = value
        return [int(c) for c in child]

    def crossover(
        self, a: VRPTWSolution, b: VRPTWSolution, rng: np.random.Generator
    ) -> tuple[VRPTWSolution, VRPTWSolution]:
        c1 = VRPTWSolution(perm=tuple(self._order_crossover(a.perm, b.perm, rng)),
                           k=a.k if rng.random() < 0.5 else b.k)
        c2 = VRPTWSolution(perm=tuple(self._order_crossover(b.perm, a.perm, rng)),
                           k=b.k if rng.random() < 0.5 else a.k)
        return c1, c2

    def mutate(self, individual: VRPTWSolution, rng: np.random.Generator) -> VRPTWSolution:
        """施加一次重排变异，并偶尔微调车辆数。

        重排变异（交换、移动、逆序）是排列编码路径问题的标准变异算子。
        """
        perm = list(individual.perm)
        n = len(perm)
        k = individual.k

        if n >= 2:
            i, j = sorted(int(v) for v in rng.choice(n, size=2, replace=False))
            move = int(rng.integers(0, 3))
            if move == 0:  # 交换
                perm[i], perm[j] = perm[j], perm[i]
            elif move == 1:  # 移动
                perm.insert(i, perm.pop(j))
            else:  # 逆序
                perm[i : j + 1] = perm[i : j + 1][::-1]

        if rng.random() < 0.25:
            delta = -1 if rng.random() < 0.5 else 1
            k = int(np.clip(k + delta, 1, self.max_routes))

        return VRPTWSolution(perm=tuple(perm), k=k)

    # -- 报告 --------------------------------------------------------------

    def describe(self, individual: VRPTWSolution) -> dict[str, Any]:
        """已解码解的完整指标记录，用于 CSV 输出。"""
        if not individual.decoded:
            self.evaluate(individual)
        return {
            "routes": ["-".join(str(c) for c in route) for route in individual.routes],
            **{name: individual.metrics.get(name) for name in OBJECTIVE_NAMES},
            "objective_vector": [individual.metrics[name] for name in self.objectives],
        }
