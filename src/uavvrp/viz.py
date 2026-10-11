"""NSGA-II 基线的绘图模块：帕累托前沿图与收敛曲线。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # 无界面后端：只把图片写盘，不弹窗

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

__all__ = ["plot_pareto_front", "plot_convergence", "plot_front_grid", "OBJECTIVE_LABELS"]

# 目标量的中文显示名，用于坐标轴与图例。
# 注意：键名是程序内部标识（distance/vehicles/makespan/waiting），不能翻译。
OBJECTIVE_LABELS = {
    "distance": "总行驶距离",
    "vehicles": "使用车辆数",
    "makespan": "最长路径时长（完工时间）",
    "waiting": "总等待时间",
}

# 按优先级排列的中文字体候选。图表标题、坐标轴、图例都要渲染中文，
# 而 matplotlib 自带的 DejaVu Sans 不含汉字，不做处理会显示成方框。
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "DengXian",
    "Microsoft JhengHei",
    "SimSun",
    "WenQuanYi Zen Hei",
)


def _configure_cjk_font() -> str | None:
    """挑选一个系统里真实存在的中文字体，返回字体名（找不到则返回 None）。

    无论是否找到中文字体，都要关掉 ``axes.unicode_minus``：否则负号会用特殊
    字符渲染，在中文字体下常常变成方框。
    """
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_FONT_CANDIDATES if name in available), None)
    if chosen is not None:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    return chosen


CJK_FONT = _configure_cjk_font()
if CJK_FONT is None:  # pragma: no cover - 取决于运行环境
    import warnings

    warnings.warn(
        "未找到可用的中文字体，图表中的中文可能显示为方框。"
        f"已尝试：{', '.join(CJK_FONT_CANDIDATES)}",
        RuntimeWarning,
        stacklevel=2,
    )


def _label(name: str) -> str:
    """把 objective 的内部标识换成中文显示名。"""
    return OBJECTIVE_LABELS.get(name, name)


def plot_pareto_front(
    front: np.ndarray,
    objectives: Sequence[str],
    path: Path | str,
    *,
    title: str,
    reference: tuple[float, ...] | None = None,
    reference_label: str = "继承的参考解（上界）",
    all_points: np.ndarray | None = None,
) -> Path:
    """画出帕累托前沿散点图，并叠加阶梯折线勾勒支配边界。

    ``all_points`` 可选地叠加各次运行最终种群的全部解（浅灰），用来展示搜索
    在目标空间里的覆盖范围。
    """
    x, y = objectives[0], objectives[1]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)

    if all_points is not None and len(all_points):
        ax.scatter(
            all_points[:, 0], all_points[:, 1],
            s=14, c="#b8b8b8", alpha=0.45, linewidths=0,
            label=f"最终种群（{len(all_points)} 个解）", zorder=1,
        )

    if len(front):
        order = np.argsort(front[:, 0], kind="stable")
        ordered = front[order]
        # 阶梯折线：勾勒被支配区域的边界。
        step_x = np.concatenate([[ordered[0, 0]], ordered[:, 0], [ordered[-1, 0]]])
        step_y = np.concatenate([[ordered[0, 1]], ordered[:, 1], [ordered[-1, 1]]])
        ax.step(step_x, step_y, where="post", color="#1f77b4", lw=1.3, alpha=0.55, zorder=2)
        ax.scatter(
            front[:, 0], front[:, 1],
            s=52, c="#1f77b4", edgecolors="white", linewidths=0.7,
            label=f"帕累托前沿（{len(front)} 个非支配解）", zorder=3,
        )
        for fx, fy in ordered:
            ax.annotate(
                f"({fx:g}, {fy:g})", (fx, fy),
                textcoords="offset points", xytext=(6, 5), fontsize=7.5, color="#14507d",
            )

    if reference is not None:
        ax.scatter(
            [reference[0]], [reference[1]],
            marker="*", s=210, c="#d62728", edgecolors="white", linewidths=0.8,
            label=reference_label, zorder=4,
        )

    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    ax.set_title(title)
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_convergence(
    history_by_seed: list[list[dict]],
    path: Path | str,
    *,
    title: str,
    metric: str = "hypervolume",
) -> Path:
    """画出各代收敛曲线：多个种子的中位数曲线 + 四分位带。"""
    path = Path(path)
    series = []
    for history in history_by_seed:
        values = [record.get(metric) for record in history]
        if any(v is None for v in values):
            series = []
            break
        series.append(np.asarray(values, dtype=float))
    if not series:
        series = [
            np.asarray([r["front_size"] for r in history], dtype=float)
            for history in history_by_seed
        ]
        metric = "front_size"

    stacked = np.vstack(series)  # 形状：(种子数, 代数)
    generations = np.arange(stacked.shape[1])
    median = np.median(stacked, axis=0)
    q1 = np.percentile(stacked, 25, axis=0)
    q3 = np.percentile(stacked, 75, axis=0)

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    ax.fill_between(generations, q1, q3, color="#1f77b4", alpha=0.18,
                    label=f"四分位区间（{len(series)} 个种子）")
    ax.plot(generations, median, color="#1f77b4", lw=1.8, label="中位数")
    for curve in stacked:
        ax.plot(generations, curve, color="#1f77b4", lw=0.5, alpha=0.28)
    ax.set_xlabel("代数")
    ax.set_ylabel("超体积" if metric == "hypervolume" else "帕累托前沿规模")
    ax.set_title(title)
    ax.grid(alpha=0.28, linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_front_grid(
    panels: list[dict[str, Any]],
    path: Path | str,
    *,
    title: str = "",
) -> Path:
    """把多个算例的帕累托前沿并排画成一张图。

    每个 panel 字典提供 ``front``、``objectives``、``title``，可选
    ``reference`` 与 ``reference_label``。
    """
    n = len(panels)
    if n == 0:
        raise ValueError("没有可绘制的面板")
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), dpi=160, squeeze=False)
    for ax, panel in zip(axes[0], panels):
        front = np.asarray(panel["front"], dtype=float)
        objectives = panel["objectives"]
        if len(front):
            order = np.argsort(front[:, 0], kind="stable")
            ordered = front[order]
            ax.step(ordered[:, 0], ordered[:, 1], where="post",
                    color="#1f77b4", lw=1.2, alpha=0.55)
            ax.scatter(front[:, 0], front[:, 1], s=46, c="#1f77b4",
                       edgecolors="white", linewidths=0.7, zorder=3)
        reference = panel.get("reference")
        if reference is not None:
            ax.scatter([reference[0]], [reference[1]], marker="*", s=170,
                       c="#d62728", edgecolors="white", linewidths=0.8, zorder=4)
        ax.set_xlabel(_label(objectives[0]))
        ax.set_ylabel(_label(objectives[1]))
        ax.set_title(panel["title"])
        ax.grid(alpha=0.28, linewidth=0.6)
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
