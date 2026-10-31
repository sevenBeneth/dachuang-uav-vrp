"""多无人机跨江配送的建模、优化与实验工具。"""

__version__ = "0.2.0"

from .solomon import Instance, Solution, parse_vrp, parse_sol, route_distance

__all__ = [
    "__version__",
    "Instance",
    "Solution",
    "parse_vrp",
    "parse_sol",
    "route_distance",
]
