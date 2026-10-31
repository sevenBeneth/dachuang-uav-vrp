"""从已保存运行生成论文所需图表，不进行二次求解。"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.indicators.hv import HV

from uavvrp.viz import CJK_FONT
from uavvrp.drone import DroneProblem, fuzzy_selection
from uavvrp.metrics import nondominated


def save(fig, path):
    is_3d = any(hasattr(ax, 'zaxis') for ax in fig.axes)
    if is_3d:
        # mplot3d 的 tight bbox 可能漏计伸出轴框的 z 标签。
        fig.subplots_adjust(left=.01, right=.86, bottom=.08, top=.91)
    fig.savefig(path, dpi=160, bbox_inches=None if is_3d else 'tight')
    plt.close(fig)


def plot_study(records, out, config_dir):
    from run_study import load_case
    figures = out/'figures'
    figures.mkdir(exist_ok=True)
    metrics = pd.read_csv(out/'metrics.csv')
    for name in sorted({r['instance'] for r in records}):
        group = [r for r in records if r['instance'] == name and r['multiplier'] == 1 and r['front']]
        if not group:
            continue
        algorithms = sorted({r['algorithm'] for r in group})
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        for algorithm in algorithms:
            front = nondominated(np.vstack([r['front'] for r in group if r['algorithm'] == algorithm]))
            ax.scatter(*front.T, label=algorithm, s=15, alpha=.7)
        ax.set(xlabel='时间目标', ylabel='闭合总航程', zlabel='累计风险', title=f'{name} 三目标非支配并集')
        ax.legend(fontsize=8)
        save(fig, figures/f'{name}_pareto.png')

        case_metrics = metrics[metrics.instance == name]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, column, label in zip(axes, ('hv', 'seconds'), ('归一化超体积', '求解墙钟时间（秒）')):
            ax.boxplot([case_metrics[case_metrics.algorithm == a][column] for a in algorithms], tick_labels=algorithms)
            ax.tick_params(axis='x', rotation=25, labelsize=8)
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
        fig.suptitle(f'{name} 各次独立运行')
        fig.tight_layout()
        save(fig, figures/f'{name}_statistics.png')

        # 整段收敛过程用同一批经验尺度，包含初期较差点，避免参考点越界。
        frames = [np.asarray(h['front']) for r in group for h in r['history'] if h.get('front')]
        if frames:
            pooled = np.vstack(frames)
            pooled = pooled[np.all(pooled < 1e12, axis=1)]
            if len(pooled):
                lower, span = pooled.min(axis=0), np.maximum(np.ptp(pooled, axis=0), 1e-9)
                indicator = HV(ref_point=np.full(3, 1.1))
                fig, ax = plt.subplots(figsize=(8, 4.5))
                convergence_rows = []
                for algorithm in algorithms:
                    curves = []
                    for r in group:
                        if r['algorithm'] != algorithm:
                            continue
                        curve = []
                        for h in r['history']:
                            front = np.asarray(h.get('front', [])).reshape(-1, 3)
                            front = front[np.all(front < 1e12, axis=1)]
                            curve.append(float(indicator((front-lower)/span)) if len(front) else 0)
                        curves.append(curve)
                    curves = np.asarray(curves)
                    x = np.arange(curves.shape[1])
                    median, q1, q3 = np.median(curves, axis=0), np.percentile(curves, 25, axis=0), np.percentile(curves, 75, axis=0)
                    line, = ax.plot(x, median, label=algorithm)
                    ax.fill_between(x, q1, q3, color=line.get_color(), alpha=.13)
                    convergence_rows.extend(dict(algorithm=algorithm, generation=int(g), hv_median=m, hv_q1=a, hv_q3=b)
                                            for g, m, a, b in zip(x, median, q1, q3))
                ax.set(xlabel='繁殖代数', ylabel='超体积（过程共同尺度）', title=f'{name} 收敛中位数与四分位区间')
                ax.legend(fontsize=8)
                ax.grid(alpha=.2)
                save(fig, figures/f'{name}_convergence.png')
                pd.DataFrame(convergence_rows).to_csv(out/f'{name}_convergence.csv', index=False)
                (out/f'{name}_convergence_scale.json').write_text(json.dumps(dict(lower=lower.tolist(), span=span.tolist(), reference=[1.1]*3)), encoding='utf-8')

        improved = [r for r in group if r['algorithm'] == 'ALNS_NSGA2']
        if improved:
            # 全部种子的非支配并集中选折中解，不固定取首个种子。
            lookup = {tuple(vector): (r, idx) for r in improved for idx, vector in enumerate(r['front'])}
            front = nondominated(np.asarray(list(lookup)))
            chosen, scores = fuzzy_selection(front)
            record, index = lookup[tuple(front[chosen])]
            solution = record['solutions'][index]
            base, size = name.rsplit('_n', 1)
            instance, scenario = load_case(base, int(size), config_dir)
            problem = DroneProblem(instance, scenario)
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            snapshots = {}
            for ax, snapshot_time in zip(axes, (0, scenario.risk_period/4)):
                xx, yy, risk = problem.network.risk_map(snapshot_time)
                im = ax.pcolormesh(xx, yy, risk, cmap='YlOrRd', vmin=0, vmax=1, shading='auto')
                ax.set(title=f't = {snapshot_time:g}', xlabel='x', ylabel='y')
                snapshots[f'risk_t{snapshot_time:g}'] = risk
            fig.colorbar(im, ax=axes, label='归一化瞬时风险', shrink=.85)
            fig.suptitle(f'{name} 动态风险快照（留白为禁飞区）')
            save(fig, figures/f'{name}_risk.png')
            np.savez_compressed(out/f'{name}_risk_maps.npz', x=xx, y=yy, **snapshots)
            output = dict(seed=record['seed'], scores=scores.tolist(), **solution)
            (out/f'{name}_selected.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axhspan(problem.network.low, problem.network.high, color='#d5e9ef', label='禁飞水域')
            for x in scenario.bridge_x:
                ax.plot([x, x], [problem.network.low, problem.network.high], color='#555555', lw=5)
            coords = np.asarray(instance.coords)
            ax.scatter(*coords[1:].T, c='#555555', s=22, zorder=4)
            ax.scatter(*coords[0], marker='*', c='#d18a20', s=180, zorder=5, label='起降点')
            for c, point in enumerate(coords[1:], 1):
                ax.annotate(str(c), point, fontsize=7, xytext=(3, 3), textcoords='offset points')
            for ri, route in enumerate(solution['routes']):
                path = []
                for a, b in zip([0]+route, route+[0]):
                    path.extend(problem.network.paths[a][b])
                path = np.asarray(path)
                ax.plot(*path.T, lw=1.3, label=f'无人机 {ri+1}', alpha=.85)
            ax.set(title=f'{name} 等权隶属度选出的闭合配送方案', xlabel='x（算例单位）', ylabel='y（算例单位）')
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(fontsize=8, loc='best')
            save(fig, figures/f'{name}_routes.png')

    sensitivity = pd.read_csv(out/'sensitivity.csv') if (out/'sensitivity.csv').stat().st_size > 1 else pd.DataFrame()
    if len(sensitivity) and sensitivity.multiplier.nunique() > 1:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        for ax, column, title in zip(axes, ('time', 'distance', 'risk'), ('时间目标', '总航程', '累计风险')):
            for name, group in sensitivity.groupby('instance'):
                grouped = group.groupby('multiplier')[column]
                med = grouped.median()
                ax.plot((med.index-1)*100, med, 'o-', label=name)
                ax.fill_between((med.index-1)*100, grouped.quantile(.25), grouped.quantile(.75), alpha=.12)
            ax.set(xlabel='最大航程扰动（%）', ylabel=title)
            ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        save(fig, figures/'battery_sensitivity.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parents[1]/'configs')
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding='utf-8')) for p in (args.out/'runs').glob('*.json')]
    plot_study(records, args.out, args.config_dir)
