"""共享 fixture：定位基准数据，并构造小规模合成实例。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from uavvrp.solomon import Instance  # noqa: E402

RAW_DIR = ROOT / "data" / "raw" / "solomon"
SUBSET_DIR = ROOT / "data" / "processed" / "subsets"

requires_raw = pytest.mark.skipif(
    not (RAW_DIR / "C101.vrp").exists(),
    reason="尚未获取 Solomon 数据；请运行 scripts/fetch_solomon.py",
)
requires_subsets = pytest.mark.skipif(
    not (SUBSET_DIR / "C101_n10.json").exists(),
    reason="尚未构建子实例；请运行 scripts/run_preprocess.py",
)


@pytest.fixture(scope="session")
def raw_dir() -> Path:
    return RAW_DIR


def make_instance(
    name: str = "TOY",
    coords: list[tuple[float, float]] | None = None,
    demand: list[int] | None = None,
    ready: list[float] | None = None,
    due: list[float] | None = None,
    service: list[float] | None = None,
    capacity: int = 100,
    vehicle_limit: int = 10,
) -> Instance:
    """手工构造的小规模实例；第一个节点为车场。"""
    coords = coords or [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 0.0)]
    n = len(coords)
    demand = demand if demand is not None else [0] + [10] * (n - 1)
    ready = ready if ready is not None else [0.0] * n
    due = due if due is not None else [1000.0] * n
    service = service if service is not None else [0.0] + [1.0] * (n - 1)
    return Instance(
        name=name,
        capacity=capacity,
        vehicle_limit=vehicle_limit,
        edge_weight_type="EUC_2D",
        coords=tuple(coords),
        demand=tuple(demand),
        ready=tuple(ready),
        due=tuple(due),
        service=tuple(service),
    )


def load_subset(name: str) -> tuple[Instance, dict]:
    payload = json.loads((SUBSET_DIR / f"{name}.json").read_text(encoding="utf-8"))
    return Instance.from_dict(payload), payload
