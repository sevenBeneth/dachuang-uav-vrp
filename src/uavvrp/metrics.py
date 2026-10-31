"""共同尺度上的 HV、经验参考集 IGD、Spacing 和配对检验。"""

import numpy as np
from pymoo.indicators.hv import HV
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon

from .nsga2 import fast_non_dominated_sort


def nondominated(F):
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError("目标矩阵必须为二维")
    F = F[np.isfinite(F).all(axis=1) & (F < 1e12).all(axis=1)]
    if not len(F):
        return F
    F = np.unique(F, axis=0)
    return F[fast_non_dominated_sort(F)[0][0]]


def indicators(front, reference, lower, span):
    front = np.asarray(front, dtype=float)
    if not len(front):
        return dict(hv=0.0, igd=None, spacing=None)
    f = (front-lower)/span
    r = (reference-lower)/span
    hv = float(HV(ref_point=np.full(front.shape[1], 1.1))(f))
    igd = float(cdist(r, f).min(axis=1).mean())
    if len(f) < 2:
        spacing = 0.0
    else:
        distances = cdist(f, f, metric='cityblock')
        np.fill_diagonal(distances, np.inf)
        spacing = float(np.std(distances.min(axis=1), ddof=1))
    return dict(hv=hv, igd=igd, spacing=spacing)


def paired_test(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) != len(b) or not len(a):
        raise ValueError("配对样本数量必须相等且非零")
    delta = a-b
    if np.all(np.abs(delta) < 1e-12):
        return 1.0
    return float(wilcoxon(delta, alternative='two-sided', method='auto').pvalue)
