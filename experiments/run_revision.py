"""扩展规模与 ALNS 触发范围对比；逐次保存进程 CPU 时间。"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'experiments')]
import numpy as np
import pandas as pd
import run_study
from run_study import solve, load_case, write_json, analyze
from compiled_decoder import CompiledDroneProblem as DroneProblem, warmup
run_study.DroneProblem = DroneProblem
from uavvrp.improved import ALNSNSGA2
from uavvrp.nsga2 import fast_non_dominated_sort


def timed_solve(job):
    warmup()
    start = time.process_time()
    record = solve(job)
    record['cpu_seconds'] = time.process_time()-start
    return record


def timing_solve(seed, full):
    warmup()
    inst, scenario = load_case('R101', 25, ROOT/'configs')
    problem = DroneProblem(inst, scenario)
    engine = ALNSNSGA2(problem, pop_size=60, n_gen=150, seed=seed, trigger_all=full)
    cpu, wall = time.process_time(), time.perf_counter()
    result = engine.run()
    cpu, wall = time.process_time()-cpu, time.perf_counter()-wall
    feasible = [i for i,s in enumerate(result.population) if s.decoded]
    indices = fast_non_dominated_sort(result.objectives[feasible])[0][0]
    chosen = [feasible[i] for i in indices]
    return dict(instance=inst.name, seed=seed, algorithm='ALL_ALNS' if full else 'ELITE_ALNS',
                cpu_seconds=cpu, seconds=wall, evaluations=result.n_evaluations,
                alns_calls=engine.calls, front=result.objectives[chosen].tolist(),
                solutions=[problem.describe(result.population[i]) for i in chosen],
                scenario=scenario.__dict__, status='ok' if chosen else 'no_feasible_solution')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scale','timing']);p.add_argument('--customers',type=int,default=50)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args();out=ROOT/'results'/('revision_timing' if a.mode=='timing' else f'revision_scale_n{a.customers}')
    warmup()
    out.mkdir(exist_ok=True);raw=out/'runs';raw.mkdir(exist_ok=True)
    configs=ROOT/'configs'/f'scale_n{a.customers}'
    manifest=dict(mode=a.mode,customers=a.customers,seeds=a.seeds,pop=60,gen=150,workers=a.workers,
                  python=platform.python_version(),cpu=platform.processor(),logical_cpus=os.cpu_count(),
                  versions={p:importlib.metadata.version(p) for p in ['numpy','numba','pymoo','scipy','pandas']},
                  timing='process_time CPU seconds; perf_counter wall seconds; single-thread numerical libraries',
                  decoder='numba compiled equivalent split; JIT warmup excluded',
                  source_hashes={f.relative_to(ROOT).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in list((ROOT/'src/uavvrp').glob('*.py'))+[Path(__file__),ROOT/'experiments/run_study.py',ROOT/'experiments/compiled_decoder.py']})
    names=['C101','R101'] if a.mode=='scale' else ['R101']
    manifest['configs']={n:json.loads(((configs if a.mode=='scale' else ROOT/'configs')/f'{n}.json').read_text()) for n in names}
    manifest['input_hashes']={n:hashlib.sha256((ROOT/'data/processed/instances'/f'{n}.json').read_bytes()).hexdigest() for n in names}
    write_json(out/'manifest.json',manifest)
    if a.mode=='scale':
        jobs=[(f'{name}_n{a.customers}__{alg}__s{seed}',(name,a.customers,str(configs),alg,seed,60,150,1.0))
              for name in ['C101','R101'] for alg in ['NSGA2','MOEAD','SPEA2','ALNS_NSGA2'] for seed in range(1,a.seeds+1)]
        with ProcessPoolExecutor(a.workers) as pool:
            futures={pool.submit(timed_solve,j):key for key,j in jobs if not (raw/f'{key}.json').exists()}
            for future in as_completed(futures):
                key=futures[future];r=future.result();write_json(raw/f'{key}.json',r)
                print(key,r['status'],round(r['cpu_seconds'],2),flush=True)
        records=[json.loads((raw/f'{key}.json').read_text()) for key,_ in jobs]
        analyze(records,out)
        pd.DataFrame([{k:r[k] for k in ['instance','algorithm','seed','cpu_seconds','seconds','evaluations','status']} for r in records]).to_csv(out/'cpu_times.csv',index=False)
    else:
        # 串行、交替执行两种方式，避免并行调度干扰触发范围对比。
        for seed in range(1,a.seeds+1):
            for full in ([False,True] if seed%2 else [True,False]):
                key=f'{"ALL" if full else "ELITE"}__s{seed}'
                if (raw/f'{key}.json').exists():continue
                r=timing_solve(seed,full);write_json(raw/f'{key}.json',r)
                print(key,round(r['cpu_seconds'],2),flush=True)
        records=[json.loads(f.read_text()) for f in raw.glob('*.json')]
        pd.DataFrame([{k:r[k] for k in ['algorithm','seed','cpu_seconds','seconds','evaluations','alns_calls','status']} for r in records]).to_csv(out/'cpu_times.csv',index=False)
    assert all(r['status']=='ok' for r in records)


if __name__=='__main__':main()
