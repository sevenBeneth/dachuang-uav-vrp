"""路径层面的 VRPTW 语义：时间、载量与成本。

"这条路径究竟做了什么" 只有一份实现，由预处理校验器、参考解检查器与 NSGA-II
解码器共用。把它集中在一处，正是整条流水线能够断言公布的 Solomon 解可被精确
复现的原因。

时间推进遵循 Solomon 的约定：

* 车辆离开车场的时间不早于 ``ready[0]``（通常为 0）；
* 车辆可以早于客户的 ready 时间到达并在此等待；
* 服务从 ``max(arrival, ready[i])`` 开始，持续 ``service[i]``；
* 晚于 ``due[i]`` 到达即为不可行；
* 返回车场的时间不得超过 ``due[0]``。

距离采用不取整的欧氏距离，这正是公布成本（例如 C101 的 ``827.3``）可以复现的
原因。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .solomon import Instance

__all__ = ["RouteStats", "simulate_route", "route_is_feasible", "route_distance"]

# 时间比较在原理上是精确的，但基准数据是十进制而浮点数不是，因此在
# "不晚于" 类检查上留一丝余量。求解器必须使用这个确切取值。
EPS = 1e-9

# 公布的 `.sol` 路径则是另一回事：56 个 Solomon 参考解中有 8 个在用精确欧氏
# 距离评估时会违反时间窗，幅度最大约 0.4 个时间单位（见 docs/data_notes.md）。
# 因此校验*这些*文件需要一个更宽松的容差；求解器绝不能使用它。
REFERENCE_TOLERANCE = 0.5


@dataclass(frozen=True)
class RouteStats:
    """一条可行路径所有可度量的信息。"""

    load: int
    distance: float
    """行驶距离：车场 -> 客户 -> 车场。"""
    duration: float
    """返回车场的时刻。即该路径对完工时间（make-span）的贡献。"""
    waiting: float
    """等待时间窗开启所花费的总时间。"""
    n_customers: int
    schedule: tuple[tuple[int, float, float], ...]
    """按顺序给出每个停靠点的 ``(customer, arrival, service_start)``。"""


def route_distance(inst: Instance, route: Sequence[int], dist: list[list[float]] | None = None) -> float:
    """``route`` 的行驶距离（车场 -> 各停靠点 -> 车场）。"""
    if not route:
        return 0.0
    matrix = dist if dist is not None else inst.distance_matrix()
    total = matrix[0][route[0]]
    for prev, nxt in zip(route, route[1:]):
        total += matrix[prev][nxt]
    total += matrix[route[-1]][0]
    return total


def simulate_route(
    inst: Instance,
    route: Sequence[int],
    dist: list[list[float]] | None = None,
    tolerance: float = EPS,
) -> RouteStats | None:
    """模拟 ``route``；返回其 :class:`RouteStats`，若不可行则返回 ``None``。

    不可行的含义是：超出载量、错过时间窗，或返回车场过晚。空路径显然可行，且
    不产生任何成本。

    ``tolerance`` 是 "不晚于" 类检查所允许的余量。求解器必须保持为
    :data:`EPS`；只有校验公布的基准文件时才应传入 :data:`REFERENCE_TOLERANCE`。
    """
    if not route:
        return RouteStats(0, 0.0, inst.ready[0], 0.0, 0, ())

    matrix = dist if dist is not None else inst.distance_matrix()
    n = inst.n_nodes

    load = 0
    distance = 0.0
    waiting = 0.0
    clock = inst.ready[0]
    prev = 0
    schedule: list[tuple[int, float, float]] = []

    for customer in route:
        if not 0 < customer < n:
            raise ValueError(f"{inst.name}: 客户编号 {customer} 超出范围 (1..{n - 1})")

        leg = matrix[prev][customer]
        distance += leg
        arrival = clock + leg
        if arrival > inst.due[customer] + tolerance:
            return None  # 错过时间窗

        start = arrival if arrival >= inst.ready[customer] else inst.ready[customer]
        waiting += start - arrival
        clock = start + inst.service[customer]

        load += inst.demand[customer]
        if load > inst.capacity:
            return None  # 车辆超载

        schedule.append((customer, arrival, start))
        prev = customer

    distance += matrix[prev][0]
    duration = clock + matrix[prev][0]
    if duration > inst.due[0] + tolerance:
        return None  # 无法按时返回车场

    return RouteStats(
        load=load,
        distance=distance,
        duration=duration,
        waiting=waiting,
        n_customers=len(route),
        schedule=tuple(schedule),
    )


def route_is_feasible(
    inst: Instance,
    route: Sequence[int],
    dist: list[list[float]] | None = None,
    tolerance: float = EPS,
) -> bool:
    return simulate_route(inst, route, dist, tolerance) is not None
