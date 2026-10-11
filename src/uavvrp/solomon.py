"""Solomon (1987) VRPTW 基准数据格式的解析。

每个算例由两个文件组成：

``<name>.vrp``
    问题数据，采用 SINTEF 与 VRP-REP 沿用的、源自 TSPLIB 的 "VRPLIB" 布局::

        NAME : C101
        TYPE : CVRPTW
        DIMENSION : 101          # 节点总数，含车场
        VEHICLES : 25            # 车队规模上限
        CAPACITY : 200
        SERVICE_TIME : 90
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
        1 40 50                  # 从 1 开始的节点编号, x, y（节点 1 为车场）
        ...
        DEMAND_SECTION
        1 0                      # 节点 1（车场）的需求量为 0
        ...
        TIME_WINDOW_SECTION
        1 0 1236                 # ready time（时间窗下界）, due time（上界）
        ...
        DEPOT_SECTION
        1
        -1
        EOF

``<name>.sol``
    参考解（当前已知最优解）::

        Route #1: 5 3 7 8 10 11 9 6 4 2 1 75
        ...
        Cost 827.3

    路径中的条目是**客户**编号，即节点编号减一，因为车场（节点 1）总是作为每条
    路径的起点与终点单独存放。

距离遵循 Solomon 的约定：平面上的普通欧氏距离，不做取整。正是这一点让 C101 等
算例公布的 ``Cost`` 值（如 ``827.3``）可以复现，而 :func:`route_distance`
也针对算例集中的每一个参考解做了回归测试。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

__all__ = [
    "Instance",
    "Solution",
    "parse_vrp",
    "parse_sol",
    "route_distance",
    "build_distance_matrix",
    "portable_path",
    "project_root",
]


def project_root() -> os.PathLike[str] | None:
    """项目根目录：包含 ``pyproject.toml`` 的那一级；找不到则返回 None。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return None


def portable_path(path: str | os.PathLike[str]) -> str:
    """把路径写成与机器无关的形式，供序列化输出使用。

    写入 ``data/processed`` 的 JSON 会带上源文件的路径。如果直接写绝对路径，
    换一台机器、换一个检出目录，重新跑一次预处理就会产生一整片无意义的 diff。
    因此这里统一输出相对于项目根目录的 POSIX 风格路径；源文件不在项目内时退化为
    文件名。
    """
    if not path:
        return ""
    candidate = Path(path)
    root = project_root()
    if root is not None:
        try:
            return candidate.resolve().relative_to(Path(root)).as_posix()
        except ValueError:
            pass
    return candidate.name

# 用于终止头部块的段落关键字。
_SECTION_KEYWORDS = {
    "NODE_COORD_SECTION",
    "DEMAND_SECTION",
    "TIME_WINDOW_SECTION",
    "DEPOT_SECTION",
    "SERVICE_TIME_SECTION",
    "EDGE_WEIGHT_SECTION",
    "EOF",
}


@dataclass(frozen=True)
class Instance:
    """不可变的 VRPTW 算例，车场位于下标 0。

    ``coords``、``demand``、``ready``、``due`` 和 ``service`` 均以内部从 0 开始
    的节点下标索引，且下标 0 恒为车场。解析器会把基准数据从 1 开始的编号以及
    车场的任意存放位置归一化掉，因此下游代码无需再关心这些细节。
    """

    name: str
    capacity: int
    vehicle_limit: int
    edge_weight_type: str
    coords: tuple[tuple[float, float], ...]
    demand: tuple[int, ...]
    ready: tuple[float, ...]
    due: tuple[float, ...]
    service: tuple[float, ...]
    source_path: str = ""

    # ---- 派生属性 / 便捷方法 ---------------------------------------------

    @property
    def n_nodes(self) -> int:
        return len(self.coords)

    @property
    def n_customers(self) -> int:
        """客户数量，即除车场之外的所有节点。"""
        return len(self.coords) - 1

    @property
    def customers(self) -> tuple[int, ...]:
        return tuple(range(1, self.n_nodes))

    @property
    def total_demand(self) -> int:
        return int(sum(self.demand))

    @property
    def horizon(self) -> float:
        """车场关闭时间：最晚可能的返回时刻。"""
        return float(self.due[0])

    @property
    def min_vehicles_by_capacity(self) -> int:
        """由总需求导出的车队规模下界。"""
        if self.capacity <= 0:
            return 0
        return math.ceil(self.total_demand / self.capacity)

    def distance(self, i: int, j: int) -> float:
        (xi, yi), (xj, yj) = self.coords[i], self.coords[j]
        return math.hypot(xi - xj, yi - yj)

    def distance_matrix(self) -> list[list[float]]:
        return build_distance_matrix(self)

    # ---- 序列化 / 反序列化 -----------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_nodes": self.n_nodes,
            "n_customers": self.n_customers,
            "capacity": self.capacity,
            "vehicle_limit": self.vehicle_limit,
            "edge_weight_type": self.edge_weight_type,
            "horizon": self.horizon,
            "total_demand": self.total_demand,
            "min_vehicles_by_capacity": self.min_vehicles_by_capacity,
            "source_path": portable_path(self.source_path),
            # 节点记录，车场在前
            "nodes": [
                {
                    "id": i,
                    "x": self.coords[i][0],
                    "y": self.coords[i][1],
                    "demand": self.demand[i],
                    "ready": self.ready[i],
                    "due": self.due[i],
                    "service": self.service[i],
                }
                for i in range(self.n_nodes)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Instance":
        nodes = sorted(payload["nodes"], key=lambda r: r["id"])
        ids = [r["id"] for r in nodes]
        if ids != list(range(len(nodes))):
            raise ValueError("节点编号必须连续且从 0 开始")
        return cls(
            name=payload["name"],
            capacity=int(payload["capacity"]),
            vehicle_limit=int(payload["vehicle_limit"]),
            edge_weight_type=payload["edge_weight_type"],
            coords=tuple((float(r["x"]), float(r["y"])) for r in nodes),
            demand=tuple(int(r["demand"]) for r in nodes),
            ready=tuple(float(r["ready"]) for r in nodes),
            due=tuple(float(r["due"]) for r in nodes),
            service=tuple(float(r["service"]) for r in nodes),
            source_path=payload.get("source_path", ""),
        )

    def subset(self, customers: Sequence[int], *, name: str | None = None) -> "Instance":
        """由选定的客户编号（从 1 开始的节点编号）构造子算例。

        车场重新编号为 0，所选客户重新编号为 1..n，因此结果是一个自包含的算例，
        任何求解器都能直接使用，无需知道它来自一个更大的基准算例。
        """
        selected = list(customers)
        if not selected:
            raise ValueError("子算例至少需要一个客户")
        if len(set(selected)) != len(selected):
            raise ValueError("子算例的客户编号必须互不重复")
        for c in selected:
            if c == 0:
                raise ValueError("车场（编号 0）是隐含的，不能列出")
            if not 0 <= c < self.n_nodes:
                raise ValueError(f"客户编号 {c} 超出 {self.name} 的范围")

        order = [0] + selected
        return Instance(
            name=name or f"{self.name}_n{len(selected)}",
            capacity=self.capacity,
            vehicle_limit=self.vehicle_limit,
            edge_weight_type=self.edge_weight_type,
            coords=tuple(self.coords[i] for i in order),
            demand=tuple(self.demand[i] for i in order),
            ready=tuple(self.ready[i] for i in order),
            due=tuple(self.due[i] for i in order),
            service=tuple(self.service[i] for i in order),
            source_path=self.source_path,
        )


@dataclass
class Solution:
    """一组路径，使用从 1 开始的客户编号（0 为隐含的车场）。"""

    name: str
    routes: list[list[int]] = field(default_factory=list)
    cost: float | None = None
    source_path: str = ""

    @property
    def n_vehicles(self) -> int:
        return len(self.routes)

    @property
    def n_customers(self) -> int:
        return sum(len(r) for r in self.routes)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_vehicles": self.n_vehicles,
            "n_customers": self.n_customers,
            "cost": self.cost,
            "routes": self.routes,
        }


# ---------------------------------------------------------------------------
# 距离辅助函数
# ---------------------------------------------------------------------------


def build_distance_matrix(inst: Instance) -> list[list[float]]:
    """完整的成对欧氏距离矩阵（不取整，遵循 Solomon 风格）。

    Solomon 算例集只使用 ``EUC_2D``；其他类型一律显式报错拒绝，而不是悄然算出
    错误的距离。
    """
    if inst.edge_weight_type != "EUC_2D":
        raise ValueError(
            f"{inst.name}: 不支持的 EDGE_WEIGHT_TYPE {inst.edge_weight_type!r} "
            "(仅支持 EUC_2D)"
        )
    n = inst.n_nodes
    pts = inst.coords
    return [
        [math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) for j in range(n)]
        for i in range(n)
    ]


def route_distance(inst: Instance, route: Iterable[int]) -> float:
    """行驶距离：车场 -> 客户（按顺序）-> 车场。"""
    seq = [0, *route, 0]
    return sum(inst.distance(seq[k], seq[k + 1]) for k in range(len(seq) - 1))


# ---------------------------------------------------------------------------
# .vrp 解析
# ---------------------------------------------------------------------------


def _strip_comment(line: str) -> str:
    """去掉 ``.vrp`` 头部/段落行末尾的 ``#`` 注释。

    不用于 ``.sol`` 文件：其路径行中合法地含有 ``#``（``Route #1: ...``），
    在那里做剥离会悄然删掉每一条路径。
    """
    return line.split("#", 1)[0].strip()


def parse_vrp(path: str | os.PathLike[str]) -> Instance:
    """把 Solomon 风格的 ``.vrp`` 文件解析为 :class:`Instance`。"""
    path = os.fspath(path)
    header: dict[str, str] = {}
    coords: dict[int, tuple[float, float]] = {}
    demands: dict[int, int] = {}
    windows: dict[int, tuple[float, float]] = {}
    services: dict[int, float] = {}
    depots: list[int] = []

    section: str | None = None
    with open(path, "r", encoding="utf-8", errors="strict") as fh:
        for raw in fh:
            line = _strip_comment(raw)
            if not line:
                continue
            token = line.split()[0].upper()

            if token in _SECTION_KEYWORDS:
                section = token
                if token == "EOF":
                    break
                continue

            if ":" in line and section is None:
                key, _, value = line.partition(":")
                header[key.strip().upper()] = value.strip()
                continue

            parts = line.split()
            if section == "NODE_COORD_SECTION":
                node = int(parts[0])
                coords[node] = (float(parts[1]), float(parts[2]))
            elif section == "DEMAND_SECTION":
                demands[int(parts[0])] = int(float(parts[1]))
            elif section == "TIME_WINDOW_SECTION":
                windows[int(parts[0])] = (float(parts[1]), float(parts[2]))
            elif section == "SERVICE_TIME_SECTION":
                services[int(parts[0])] = float(parts[1])
            elif section == "DEPOT_SECTION":
                value = int(parts[0])
                if value == -1:
                    section = None
                else:
                    depots.append(value)
            else:
                # 不属于任何段落的未知游离行：为稳妥起见直接忽略。
                continue

    name = header.get("NAME", os.path.splitext(os.path.basename(path))[0])
    if not coords:
        raise ValueError(f"{path}: NODE_COORD_SECTION 缺失或为空")
    if not demands:
        raise ValueError(f"{path}: DEMAND_SECTION 缺失或为空")
    if not windows:
        raise ValueError(f"{path}: TIME_WINDOW_SECTION 缺失或为空")

    declared_dim = int(header.get("DIMENSION", len(coords)))
    if len(coords) != declared_dim:
        raise ValueError(
            f"{name}: DIMENSION={declared_dim}，但实际找到 {len(coords)} 个节点坐标"
        )
    if len(demands) != declared_dim:
        raise ValueError(
            f"{name}: DIMENSION={declared_dim}，但实际找到 {len(demands)} 条需求"
        )
    if len(windows) != declared_dim:
        raise ValueError(
            f"{name}: DIMENSION={declared_dim}，但实际找到 {len(windows)} 个时间窗"
        )

    if not depots:
        raise ValueError(f"{name}: DEPOT_SECTION 未声明车场")
    if len(depots) > 1:
        raise ValueError(f"{name}: 声明了 {len(depots)} 个车场，应为恰好 1 个")
    depot_id = depots[0]

    default_service = float(header.get("SERVICE_TIME", 0.0))
    # 节点顺序：车场在前，其余节点按编号升序排列。
    order = [depot_id] + [i for i in sorted(coords) if i != depot_id]

    def service_of(node_id: int) -> float:
        """节点的服务时长。

        Solomon 的全局 ``SERVICE_TIME`` 是*客户*的服务时长；车场没有服务时间，
        因为车辆装完货即出发。若存在逐节点的 ``SERVICE_TIME_SECTION``，对车场
        而言仍以该段落的取值为准。
        """
        if node_id in services:
            return float(services[node_id])
        return 0.0 if node_id == depot_id else default_service

    return Instance(
        name=name,
        capacity=int(float(header.get("CAPACITY", 0))),
        vehicle_limit=int(float(header.get("VEHICLES", declared_dim))),
        edge_weight_type=header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper(),
        coords=tuple(coords[i] for i in order),
        demand=tuple(demands[i] for i in order),
        ready=tuple(windows[i][0] for i in order),
        due=tuple(windows[i][1] for i in order),
        service=tuple(service_of(i) for i in order),
        source_path=path,
    )


# ---------------------------------------------------------------------------
# .sol 解析
# ---------------------------------------------------------------------------


def parse_sol(path: str | os.PathLike[str]) -> Solution:
    """解析 Solomon 风格的 ``.sol`` 文件（参考路径 + 成本）。"""
    import re

    path = os.fspath(path)
    routes: list[list[int]] = []
    cost: float | None = None
    route_re = re.compile(r"^Route\s*#\d+\s*:\s*(.*)$", re.IGNORECASE)
    cost_re = re.compile(r"^Cost\s+([0-9.]+)\s*$", re.IGNORECASE)

    with open(path, "r", encoding="utf-8", errors="strict") as fh:
        for raw in fh:
            # 这里不做注释剥离：`#` 是 `Route #N:` 语法的一部分。
            line = raw.strip()
            if not line:
                continue
            m = route_re.match(line)
            if m:
                routes.append([int(tok) for tok in m.group(1).split()])
                continue
            m = cost_re.match(line)
            if m:
                cost = float(m.group(1))
                continue

    name = os.path.splitext(os.path.basename(path))[0]
    return Solution(name=name, routes=routes, cost=cost, source_path=path)
