"""uav-vrp-nsga2-alns -- 面向多无人机配送路径问题的研究代码。

第一阶段（本次提交）提供：
  * 一个可复现的 Solomon (1987) VRPTW 基准算例加载器，
  * 一条预处理 / 校验流水线，把原始基准文件转换为经过校验、归一化的 JSON
    实例，以及固定规模的子实例，
  * 一个从零实现、教科书式的 NSGA-II 基线，并附带 Pareto 前沿报告。

第二阶段（后续）将加入本项目真正的贡献：跨江飞行约束、无人机载重 / 续航限制，
以及项目笔记中描述的 ALNS 精英解增强。
"""

__version__ = "0.1.0"

from .solomon import Instance, Solution, parse_vrp, parse_sol, route_distance

__all__ = [
    "__version__",
    "Instance",
    "Solution",
    "parse_vrp",
    "parse_sol",
    "route_distance",
]
