"""Positive controls: prove the gates release under genuine mode structure."""

import numpy as np

from . import static_gate as SG
from . import dynamic_gate as DG


def static_positive_control(seeds=range(100, 112), var_true=(0.1, 9.0),
                            N=600, J=220, n_per=60, Q=48):
    """Two subpopulations with strongly different ability variance.

    A correct mixture detector must (i) beat a single global lambda by >=1%
    held-out NLL, (ii) BIC-select R=2, and (iii) route with AUC >= 0.65.
    """
    n_fit = max(10, n_per * 2 // 3)      # leave n_per - n_fit held-out responses
    results = []
    for s in seeds:
        D = SG.generate_static(s, N=N, J=J, n_per=n_per, var_true=var_true)
        results.append(SG.audit_static(D["d"], D["items"], D["y"], n_fit=n_fit,
                                       Q=Q, z=D["z"]))
    verdict, summary = SG.static_gate(results)
    return verdict, summary, results


def dynamic_positive_control(seeds=range(200, 212), E_TR=100, E_FUT=40,
                             G_TRUE=(0.02, 0.6), DT_MEAN=0.3, Q=48):
    """Strongly separated forgetting rates over a longer span.

    Returns (design_gate, identifiability, summary) of the v2.3 DGP with fresh
    future items, so per-student routing is in principle identifiable.
    """
    runs = []
    for s in seeds:
        D = DG.generate_dynamic(s, E_TR=E_TR, E_FUT=E_FUT, G_TRUE=G_TRUE,
                                DT_MEAN=DT_MEAN, fresh_future=True)
        runs.append(DG.audit_dynamic(D, Q=Q))
    design, ident, summary = DG.dynamic_gate(runs)
    return design, ident, summary, runs
