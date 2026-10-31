"""有桥梁通道的带状禁飞区、动态航段风险与三目标无人机配送模型。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import hypot, isfinite, sin, pi, exp
from typing import Sequence

import numpy as np

from .problem import VRPTWProblem, VRPTWSolution, INFEASIBLE_OBJECTIVE
from .solomon import Instance


@dataclass(frozen=True)
class Scenario:
    river_y: float
    river_width: float
    bridge_x: tuple[float, ...]
    capacity: float = 200
    max_distance: float = 250
    n_drones: int = 6
    speed: float = 1
    max_duration: float = 1500
    early_weight: float = 1
    late_weight: float = 2
    urgent_weight: float = 5
    urgent: tuple[int, ...] = ()
    risk_amplitude: float = 0.5
    risk_period: float = 240
    risk_decay: float = 15
    obstacles: tuple[tuple[float, float], ...] = ()
    risk_samples: int = 5

    def __post_init__(self):
        positive = (self.river_width, self.capacity, self.max_distance, self.speed,
                    self.max_duration, self.risk_period, self.risk_decay)
        if not all(isfinite(v) and v > 0 for v in positive):
            raise ValueError("宽度、载重、续航、速度、风险周期和衰减长度必须为正数")
        if not isfinite(self.river_y) or not all(isfinite(x) for x in self.bridge_x):
            raise ValueError("河流与桥梁坐标必须有限")
        if self.n_drones < 1 or self.risk_samples < 1:
            raise ValueError("无人机数和风险采样数必须为正整数")
        if any(not isfinite(v) or v < 0 for v in
               (self.early_weight, self.late_weight, self.urgent_weight, self.risk_amplitude)):
            raise ValueError("惩罚权重和风险幅值必须非负")


class RiverNetwork:
    """同岸直飞，异岸仅经桥两端；无桥时两岸不连通。"""

    def __init__(self, coords, scenario: Scenario):
        self.coords = tuple(tuple(p) for p in coords)
        self.scenario = scenario
        self.low = scenario.river_y - scenario.river_width / 2
        self.high = scenario.river_y + scenario.river_width / 2
        for point in self.coords:
            if self.low < point[1] < self.high:
                raise ValueError(f"节点 {point} 位于禁飞水域内，请调整场景")
        self.paths = [[self.path(a, b) for b in self.coords] for a in self.coords]
        self.dist = [[sum(hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:]))
                      if path else float('inf') for path in row] for row in self.paths]
        self._risk_terms = [[self._quadrature(path) for path in row] for row in self.paths]

    def path(self, a, b):
        if (a[1] <= self.low and b[1] <= self.low) or (a[1] >= self.high and b[1] >= self.high):
            return (a, b)
        options = []
        for x in self.scenario.bridge_x:
            first, second = ((x, self.low), (x, self.high)) if a[1] <= self.low else ((x, self.high), (x, self.low))
            path = (a, first, second, b)
            length = sum(hypot(q[0]-p[0], q[1]-p[1]) for p, q in zip(path, path[1:]))
            options.append((length, path))
        return min(options, key=lambda item: item[0])[1] if options else ()

    def _quadrature(self, path):
        """预存中点积分的空间系数；时变项在实际出发时计算。"""
        s = self.scenario
        static_integral, offset, terms = 0.0, 0.0, []
        bound = max(1.0, .5*len(s.obstacles)+s.risk_amplitude*len(s.bridge_x))
        for a, b in zip(path, path[1:]):
            duration = hypot(b[0]-a[0], b[1]-a[1]) / s.speed
            for sample in range(s.risk_samples):
                u = (sample+.5)/s.risk_samples
                x, y = a[0] + u*(b[0]-a[0]), a[1] + u*(b[1]-a[1])
                factor = duration/s.risk_samples/bound
                static_integral += factor*sum(.5*exp(-hypot(x-ox, y-oy)/s.risk_decay) for ox, oy in s.obstacles)
                amplitude = factor*sum(s.risk_amplitude*exp(-hypot(x-bx, y-s.river_y)/s.risk_decay) for bx in s.bridge_x)
                terms.append((2*pi*(offset+u*duration)/s.risk_period, amplitude))
            offset += duration
        return static_integral, tuple(terms)

    def risk(self, i, j, departure):
        """按实际飞行时间积分；桥梁通道允许通行，不计禁飞标记值。"""
        static, terms = self._risk_terms[i][j]
        phase = 2*pi*departure/self.scenario.risk_period
        return static+sum(amplitude*max(0.0, sin(phase+offset)) for offset, amplitude in terms)

    def risk_at(self, x, y, time):
        s = self.scenario
        static = sum(.5*exp(-hypot(x-ox, y-oy)/s.risk_decay) for ox, oy in s.obstacles)
        wave = max(0.0, sin(2*pi*time/s.risk_period))
        dynamic = sum(s.risk_amplitude*wave*exp(-hypot(x-bx, y-s.river_y)/s.risk_decay)
                      for bx in s.bridge_x)
        bound = max(1.0, .5*len(s.obstacles)+s.risk_amplitude*len(s.bridge_x))
        return (static+dynamic)/bound

    def risk_map(self, time, grid_size=80):
        if grid_size < 2:
            raise ValueError("网格边长至少为 2")
        coords = np.asarray(self.coords)
        x = np.linspace(coords[:, 0].min()-5, coords[:, 0].max()+5, grid_size)
        y = np.linspace(coords[:, 1].min()-5, coords[:, 1].max()+5, grid_size)
        xx, yy = np.meshgrid(x, y)
        values = np.array([[self.risk_at(a, b, time) for a, b in zip(rx, ry)] for rx, ry in zip(xx, yy)])
        forbidden = (yy > self.low) & (yy < self.high)
        # 地图用一个网格宽表示桥；精确航路仍使用几何桥中心线。
        for bx in self.scenario.bridge_x:
            forbidden &= np.abs(xx-bx) > (x[1]-x[0])/2
        values[forbidden] = np.nan
        return xx, yy, values


class DroneProblem(VRPTWProblem):
    """时间惩罚、闭合航程、时变风险三目标，载重和续航为硬约束。

    时间窗按论文作为软约束，提前到达后等待再服务。所有无人机从车场
    ready[0] 同时出发，每架执行一次闭合航次，不含换电和跨岸调机。
    """

    def __init__(self, instance: Instance, scenario: Scenario):
        super().__init__(instance, objectives=("distance", "makespan"))
        self.scenario = scenario
        if not instance.n_customers:
            raise ValueError("至少需要一个客户")
        if any(c not in instance.customers for c in scenario.urgent):
            raise ValueError("紧急客户编号越界")
        self.network = RiverNetwork(instance.coords, scenario)
        self.dist = self.network.dist
        self.objectives = ("time", "distance", "risk")
        self.n_obj = 3
        self.max_routes = min(instance.n_customers, scenario.n_drones)
        self.nadir = np.array([1e7, scenario.max_distance*self.max_routes+1,
                               scenario.max_duration*self.max_routes+1])
        self._cached = lru_cache(maxsize=8192)(self._compute)

    def route_stats(self, route, with_risk=True):
        s, inst = self.scenario, self.instance
        if sum(inst.demand[c] for c in route) > s.capacity:
            return None
        distance, clock, objective, waiting, risk = 0.0, inst.ready[0], 0.0, 0.0, 0.0
        prev = 0
        for c in (*route, 0):
            leg = self.dist[prev][c]
            if not isfinite(leg):
                return None
            if with_risk:
                risk += self.network.risk(prev, c, clock)
            clock += leg/s.speed
            distance += leg
            if distance > s.max_distance + 1e-9:
                return None
            if c:
                early = max(0, inst.ready[c]-clock)
                late = max(0, clock-inst.due[c])
                objective += clock + s.early_weight*early + (s.urgent_weight if c in s.urgent else s.late_weight)*late
                waiting += early
                clock += early + inst.service[c]
            prev = c
        duration = clock-inst.ready[0]
        if duration > s.max_duration + 1e-9:
            return None
        return dict(time=objective, distance=distance, risk=risk, duration=duration, waiting=waiting,
                    load=sum(inst.demand[c] for c in route))

    def _segment_table(self, perm):
        n = len(perm)
        if n != self.n_customers or set(perm) != set(self.customers):
            raise ValueError("perm 必须恰好包含每个客户一次")
        table = [[float('inf')]*(n+1) for _ in range(n+1)]
        # 累计状态避免逐段重新仿真；返航始终计入续航。
        s, inst = self.scenario, self.instance
        for i in range(n):
            load, distance, clock, prev = 0, 0.0, inst.ready[0], 0
            for j in range(i+1, n+1):
                c = perm[j-1]
                load += inst.demand[c]
                distance += self.dist[prev][c]
                if load > s.capacity or not isfinite(distance):
                    break
                clock = max(clock+self.dist[prev][c]/s.speed, inst.ready[c])+inst.service[c]
                back = self.dist[c][0]
                if distance+back <= s.max_distance+1e-9 and clock+back/s.speed-inst.ready[0] <= s.max_duration+1e-9:
                    table[i][j] = distance+back
                prev = c
        return table

    def _compute(self, perm, k):
        decoded = self.decode(perm, k)
        if decoded is None:
            return (), (), 0
        routes, used = decoded
        stats = [self.route_stats(tuple(r)) for r in routes]
        if any(r is None for r in stats):
            raise AssertionError("解码和仿真约束不一致")
        metrics = {name: sum(r[name] for r in stats) for name in ("time", "distance", "risk", "waiting")}
        metrics.update(vehicles=float(used), makespan=max(r['duration'] for r in stats))
        return tuple(tuple(r) for r in routes), tuple(metrics.items()), used

    def evaluate(self, individual: VRPTWSolution):
        routes, items, used = self._cached(tuple(individual.perm), individual.k)
        individual.routes = [list(r) for r in routes]
        individual.metrics = dict(items)
        individual.decoded = bool(routes)
        if not routes:
            return np.full(3, INFEASIBLE_OBJECTIVE)
        individual.k = used
        return np.array([individual.metrics[n] for n in self.objectives])

    def describe(self, individual):
        self.evaluate(individual)
        if not individual.decoded:
            raise ValueError("无法输出不可行配送方案")
        return dict(routes=individual.routes, **individual.metrics,
                    objective_vector=[individual.metrics[n] for n in self.objectives])

    def greedy(self, kind, rng):
        if kind == 'deadline':
            perm = sorted(self.customers, key=lambda c: (self.instance.due[c], c))
        else:
            pending, perm, prev = set(self.customers), [], 0
            while pending:
                c = min(pending, key=lambda c: (self.dist[prev][c], c))
                pending.remove(c)
                perm.append(c)
                prev = c
        return VRPTWSolution(tuple(perm), int(rng.integers(1, self.max_routes+1)))


def fuzzy_selection(F, weights: Sequence[float] | None = None):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or not len(F) or not np.isfinite(F).all():
        raise ValueError("前沿必须为非空有限二维矩阵")
    w = np.ones(F.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (F.shape[1],) or not np.isfinite(w).all() or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("权重须与目标匹配、非负且总和为正")
    span = np.ptp(F, axis=0)
    membership = 1-np.divide(F-F.min(axis=0), span, out=np.zeros_like(F), where=span > 0)
    scores = membership @ (w/w.sum())
    return int(np.argmax(scores)), scores
