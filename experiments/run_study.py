"""三目标对比、消融、续航敏感性分析及 SQLite 结果归档。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sqlite3
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import pandas as pd
from uavvrp.drone import DroneProblem, Scenario, fuzzy_selection
from uavvrp.improved import ALNSNSGA2
from uavvrp.metrics import nondominated, indicators, paired_test
from uavvrp.nsga2 import NSGA2, fast_non_dominated_sort
from uavvrp.solomon import Instance


def load_case(name, size, config_dir):
    payload = json.loads((ROOT/'data/processed/instances'/f'{name}.json').read_text(encoding='utf-8'))
    original = Instance.from_dict(payload)
    if not 1 <= size <= original.n_customers:
        raise ValueError('客户数超出算例范围')
    fields = {key: getattr(original, key)[:size+1] for key in ('coords', 'demand', 'ready', 'due', 'service')}
    instance = replace(original, name=f'{name}_n{size}', **fields)
    config = json.loads((config_dir/f'{name}.json').read_text(encoding='utf-8'))
    for key in ('bridge_x', 'urgent', 'obstacles'):
        if key in config:
            config[key] = tuple(tuple(v) if isinstance(v, list) else v for v in config[key])
    config['urgent'] = tuple(c for c in config.get('urgent', ()) if c <= size)
    return instance, Scenario(**config)


def solve(job):
    name, size, config_dir, algorithm, seed, pop, gen, multiplier = job
    instance, scenario = load_case(name, size, Path(config_dir))
    scenario = replace(scenario, max_distance=scenario.max_distance*multiplier)
    problem = DroneProblem(instance, scenario)
    started = time.perf_counter()
    if algorithm in ('MOEAD', 'SPEA2'):
        from uavvrp.comparison import run_comparator
        solutions, F, history, evaluations = run_comparator(algorithm, problem, pop, gen, seed)
        operator_stats = None
    else:
        cls = NSGA2 if algorithm == 'NSGA2' else ALNSNSGA2
        options = {}
        if cls is ALNSNSGA2:
            options = dict(use_alns=algorithm != 'NO_ALNS', use_greedy=algorithm != 'NO_GREEDY',
                           use_schedule=algorithm != 'NO_SCHEDULE')
        engine = cls(problem, pop_size=pop, n_gen=gen, seed=seed, **options)
        result = engine.run()
        solutions, F, evaluations = result.population, result.objectives, result.n_evaluations
        history = result.history
        operator_stats = dict(weights=engine.weights.tolist(), usages=engine.usages.tolist(),
                              calls=engine.calls, accepted=engine.accepted) if cls is ALNSNSGA2 else None
    elapsed = time.perf_counter()-started
    feasible = [i for i, solution in enumerate(solutions) if solution.decoded]
    if not feasible:
        return dict(instance=instance.name, algorithm=algorithm, seed=seed, multiplier=multiplier,
                    front=[], solutions=[], seconds=elapsed, evaluations=evaluations,
                    history=history, operator_stats=operator_stats, scenario=asdict(scenario),
                    status='no_feasible_solution')
    feasible_F = F[feasible]
    front_indices = fast_non_dominated_sort(feasible_F)[0][0]
    unique = {}
    for idx in front_indices:
        actual = feasible[int(idx)]
        unique.setdefault(tuple(float(v) for v in F[actual]), solutions[actual])
    pairs = sorted(unique.items())
    front = np.asarray([v for v, _ in pairs])
    descriptions = [problem.describe(s) for _, s in pairs]
    selected, scores = fuzzy_selection(front)
    return dict(instance=instance.name, algorithm=algorithm, seed=seed, multiplier=multiplier,
                front=front.tolist(), solutions=descriptions, selected_index=selected,
                membership_scores=scores.tolist(), seconds=elapsed, evaluations=evaluations,
                history=history, operator_stats=operator_stats, scenario=asdict(scenario), status='ok')


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def analyze(records, out):
    rows, significance = [], []
    for instance in sorted({r['instance'] for r in records}):
        group = [r for r in records if r['instance'] == instance and r['multiplier'] == 1]
        nonempty = [np.asarray(r['front']) for r in group if r['front']]
        if not nonempty:
            continue
        pool = np.vstack(nonempty)
        reference = nondominated(pool)
        lower, span = pool.min(axis=0), np.maximum(np.ptp(pool, axis=0), 1e-9)
        write_json(out/f'{instance}_reference.json', dict(source='本批全部算法全部种子的非支配并集',
                   lower=lower.tolist(), span=span.tolist(), hv_reference=[1.1]*3, front=reference.tolist()))
        for r in group:
            metrics = indicators(r['front'], reference, lower, span)
            row = dict(instance=instance, algorithm=r['algorithm'], seed=r['seed'],
                       seconds=r['seconds'], evaluations=r['evaluations'], front_size=len(r['front']), **metrics)
            rows.append(row)
        target = sorted([r for r in rows if r['instance'] == instance and r['algorithm'] == 'ALNS_NSGA2'], key=lambda r: r['seed'])
        tests = []
        for algorithm in sorted({r['algorithm'] for r in group}-{'ALNS_NSGA2'}):
            control = sorted([r for r in rows if r['instance'] == instance and r['algorithm'] == algorithm], key=lambda r: r['seed'])
            if target and [r['seed'] for r in target] == [r['seed'] for r in control]:
                p = paired_test([r['hv'] for r in target], [r['hv'] for r in control])
                tests.append(dict(instance=instance, control=algorithm, n_pairs=len(target),
                                  hv_mean_difference=float(np.mean([a['hv']-b['hv'] for a, b in zip(target, control)])),
                                  p_value=p))
        # Holm 校正控制同算例多重检验错误率。
        running = 0.0
        for rank, row in enumerate(sorted(tests, key=lambda r: r['p_value'])):
            running = max(running, min(1, row['p_value']*(len(tests)-rank)))
            row['p_holm'] = running
            significance.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out/'metrics.csv', index=False)
    if len(metrics):
        summary = metrics.groupby(['instance', 'algorithm']).agg(
            runs=('seed', 'count'), hv_mean=('hv', 'mean'), hv_std=('hv', 'std'),
            igd_mean=('igd', 'mean'), spacing_mean=('spacing', 'mean'), seconds_mean=('seconds', 'mean'),
            evaluations_mean=('evaluations', 'mean'), front_size_mean=('front_size', 'mean')).reset_index()
        summary.to_csv(out/'summary.csv', index=False)
    pd.DataFrame(significance).to_csv(out/'significance.csv', index=False)
    sensitivity = []
    for r in records:
        if r['algorithm'] == 'ALNS_NSGA2' and r['front']:
            selected = r['solutions'][r['selected_index']]
            sensitivity.append(dict(instance=r['instance'], multiplier=r['multiplier'], seed=r['seed'],
                                    **{k: selected[k] for k in ('time', 'distance', 'risk', 'vehicles', 'makespan')}))
    pd.DataFrame(sensitivity).to_csv(out/'sensitivity.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instances', nargs='+', default=['C101', 'R101', 'RC101'])
    parser.add_argument('--customers', type=int, default=25)
    parser.add_argument('--pop', type=int, default=60)
    parser.add_argument('--gen', type=int, default=150)
    parser.add_argument('--seeds', type=int, default=30)
    parser.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--algorithms', nargs='+', choices=['NSGA2', 'MOEAD', 'SPEA2', 'ALNS_NSGA2', 'NO_ALNS', 'NO_GREEDY', 'NO_SCHEDULE'],
                        default=['NSGA2', 'MOEAD', 'SPEA2', 'ALNS_NSGA2'])
    parser.add_argument('--sensitivity', action='store_true')
    parser.add_argument('--config-dir', type=Path, default=ROOT/'configs')
    parser.add_argument('--out', type=Path, default=ROOT/'results/study')
    args = parser.parse_args(argv)
    if min(args.customers, args.gen, args.seeds, args.workers) < 1 or args.pop < 5:
        parser.error('客户数、代数、种子数和进程数须为正，种群至少为 5')
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out/'runs'
    raw.mkdir(exist_ok=True)
    manifest = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest.update(python=platform.python_version(), versions={p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'pymoo', 'pandas')},
                    timing='每次求解的墙钟时间，含局部搜索，不含进程启动与文件输出；多进程下并非独占 CPU 用时',
                    source_hashes={p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in list((ROOT/'src/uavvrp').glob('*.py'))+[Path(__file__)]+list(args.config_dir.glob('*.json'))},
                    input_hashes={name: hashlib.sha256((ROOT/'data/processed/instances'/f'{name}.json').read_bytes()).hexdigest() for name in args.instances})
    manifest_path = args.out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        parser.error('输出目录已有不同参数或源码的实验，请使用新的 --out 目录')
    write_json(manifest_path, manifest)
    jobs = [(name, args.customers, str(args.config_dir), algorithm, seed, args.pop, args.gen, 1.0)
            for name in args.instances for algorithm in args.algorithms for seed in range(1, args.seeds+1)]
    if args.sensitivity:
        jobs += [(name, args.customers, str(args.config_dir), 'ALNS_NSGA2', seed, args.pop, args.gen, multiplier)
                 for name in args.instances for multiplier in (.8, .9, 1.1, 1.2) for seed in range(1, args.seeds+1)]
    def key(job):
        return f'{job[0]}_n{job[1]}__{job[3]}__s{job[4]}__L{job[7]:.1f}'
    pending = [job for job in jobs if not (raw/f'{key(job)}.json').exists()]
    print(f'共 {len(jobs)} 次运行，待计算 {len(pending)} 次，进程数 {args.workers}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job, record in zip(pending, pool.map(solve, pending, chunksize=1)):
            write_json(raw/f'{key(job)}.json', record)
            print(f"{key(job)}: {record['status']} front={len(record['front'])} {record['seconds']:.2f}s", flush=True)
    records = [json.loads((raw/f'{key(job)}.json').read_text(encoding='utf-8')) for job in jobs]
    with sqlite3.connect(args.out/'study.sqlite3') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, instance TEXT, algorithm TEXT, seed INTEGER, multiplier REAL, payload TEXT)')
        for job, r in zip(jobs, records):
            conn.execute('INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)',
                         (key(job), r['instance'], r['algorithm'], r['seed'], r['multiplier'], json.dumps(r, ensure_ascii=False, allow_nan=False)))
    analyze(records, args.out)
    from plot_study import plot_study
    plot_study(records, args.out, args.config_dir)
    if any(r['status'] != 'ok' for r in records):
        print('存在未找到可行解的运行，已记录状态，不据此声称实验成功', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
