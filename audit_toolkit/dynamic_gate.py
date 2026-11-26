"""Longitudinal forgetting-rate boundary check (scalar forgetting readout).

Reproduces the v2.3 synthetic design: two modes with mode-specific forgetting
rate g_r, a support/transfer/future window over an exponential-arrival timeline,
and the arms D0 (static snapshot), D1 (global g), D4 (learned 2-mode mixture),
D5 (oracle mode), MixTrue (true-pi mixture). The design gate precedes the
identifiability criteria (ARI / routing AUC / g_r Pearson).
"""
import itertools
import numpy as np
from scipy.special import expit, logsumexp
from scipy.optimize import linear_sum_assignment

from .core_laplace import gh_nodes, ari, auc, bootstrap_ci, pearson

GS_DEFAULT = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]
LAMS_DEFAULT = [0.25, 0.5, 1.0, 2.0]


def generate_dynamic(seed, N=600, J=220, E_SUP=10, E_TR=20, E_FUT=10,
                     G_TRUE=(0.02, 0.6), DT_MEAN=0.3, fresh_future=False, a_scale=0.5):
    """Generate the v2.3 dynamic DGP.

    fresh_future : if True the future window draws FRESH items (predictive);
                   if False the future window REVISITS support items.
    """
    rng = np.random.default_rng(seed)
    while True:
        pi = rng.dirichlet([1, 1])
        if pi.min() >= 0.20:
            break
    a = np.log1p(np.exp(rng.normal(0, a_scale, J))) + 0.1
    d = rng.normal(0, 1, J)
    th = rng.normal(0, 1, N)
    z = rng.choice(2, N, p=pi)
    sup = np.stack([rng.choice(J, E_SUP, replace=False) for _ in range(N)])
    tr = np.stack([rng.choice(J, E_TR, replace=True) for _ in range(N)])
    if fresh_future:
        fut = np.stack([rng.choice(J, E_FUT, replace=False) for _ in range(N)])
    else:
        fut = sup
    items = np.concatenate([sup, tr, fut], 1)
    dt = rng.exponential(DT_MEAN, (N, E_SUP + E_TR + E_FUT))
    t = np.cumsum(dt, 1) - np.cumsum(dt, 1)[:, :1]
    G = np.asarray(G_TRUE, float)
    u = th[:, None] * np.exp(-G[z][:, None] * t)
    y = (rng.random((N, items.shape[1])) < expit(a[items] * u - d[items])).astype(float)
    return dict(pi=pi, a=a, d=d, z=z, items=items, t=t, y=y,
                E_SUP=E_SUP, E_TR=E_TR, E_FUT=E_FUT, G_TRUE=G)


def _ll_events(D, lam, g, Q):
    """(N,Q,E) log p(y_it | theta_q, mode g)."""
    th, logw = gh_nodes(lam, Q)
    a_i = D["a"][D["items"]]
    d_i = D["d"][D["items"]]
    eta = th[None, :, None] * np.exp(-g * D["t"])[:, None, :]
    eta = a_i[:, None, :] * eta - d_i[:, None, :]
    s = 2.0 * D["y"] - 1.0
    return -np.logaddexp(0.0, -s[:, None, :] * eta), logw


def _logm(ll, logw, idx):
    return logsumexp(logw[None, :] + ll[:, :, idx].sum(2), axis=1)


def _post(ll, logw, idx):
    s = logw[None, :] + ll[:, :, idx].sum(2)
    return np.exp(s - logsumexp(s, 1, keepdims=True))


def _fut_nll(ll, post_, idx):
    p = (post_[:, :, None] * np.exp(ll[:, :, idx])).sum(1)
    return -np.log(p).mean(1)


def audit_dynamic(D, gs=GS_DEFAULT, lams=LAMS_DEFAULT, Q=48,
                  tr_frac=0.65, tu_frac=0.15):
    """Run all arms and identifiability on one generated dynamic dataset.

    Returns a dict with per-seed metrics and gains (in %), matching the paper's
    `run_seed` output keys.
    """
    N = D["y"].shape[0]
    E_SUP, E_FUT = D["E_SUP"], D["E_FUT"]
    E = D["items"].shape[1]
    tr_end = int(N * tr_frac)
    tu_end = int(N * (tr_frac + tu_frac))
    tr = np.arange(tr_end)
    tu = np.arange(tr_end, tu_end)
    te = np.arange(tu_end, N)

    sup_idx = np.arange(E_SUP)
    fut_idx = np.arange(E - E_FUT, E)
    st_idx = np.arange(0, E - E_FUT)          # support + transfer

    # D1: choose (lam, g) by Tuning future NLL; cache st marginals for D4
    best = None
    ml_st = {}
    for lam in lams:
        for g in gs:
            ll, logw = _ll_events(D, lam, g, Q)
            ml_st[(lam, g)] = _logm(ll, logw, st_idx)
            v = _fut_nll(ll, _post(ll, logw, st_idx), fut_idx)[tu].mean()
            if best is None or v < best[0]:
                best = (v, lam, g)
    _, lam, g1 = best

    ll1, lw1 = _ll_events(D, lam, g1, Q)
    d1 = _fut_nll(ll1, _post(ll1, lw1, st_idx), fut_idx)
    ll0, lw0 = _ll_events(D, lam, 0.0, Q)
    d0 = _fut_nll(ll0, _post(ll0, lw0, sup_idx), fut_idx)   # static snapshot

    gi_true = tuple(sorted({gs.index(D["G_TRUE"][0]), gs.index(D["G_TRUE"][1])}))

    def pred_probs(llg, lwg, post_idx):
        return (_post(llg, lwg, post_idx)[:, :, None] * np.exp(llg[:, :, fut_idx])).sum(1)

    # D5 / MixTrue: oracle modes
    llt = [_ll_events(D, lam, gs[k], Q) for k in gi_true]
    pr_true = np.stack([pred_probs(llg, lwg, st_idx) for llg, lwg in llt], 1)
    d5 = -np.log((np.eye(2)[D["z"]][:, :, None] * pr_true).sum(1)).mean(1)
    mt = -np.log((np.tile(D["pi"], (N, 1))[:, :, None] * pr_true).sum(1)).mean(1)

    # D4: routing on support+transfer marginals over all (g_a, g_b) pairs
    ml_all = np.stack([ml_st[(lam, gs[k])] for k in range(len(gs))], 1)
    best4 = None
    for a, b in itertools.combinations(range(1, len(gs)), 2):
        pi, llv = _fit_pi(ml_all[tr][:, [a, b]])
        if best4 is None or llv > best4[0]:
            best4 = (llv, (a, b), pi)
    _, gi, pi = best4
    pr_modes = np.stack([pred_probs(*_ll_events(D, lam, gs[k], Q), st_idx) for k in gi], 1)
    rho = _routing(ml_all, gi, pi)
    d4 = -np.log((rho[:, :, None] * pr_modes).sum(1)).mean(1)

    # identifiability on test
    rho_tr = _routing(ml_all[tr], gi, pi)
    cost = np.array([[-(rho_tr[D["z"][tr] == c, k]).sum() for k in range(2)] for c in range(2)])
    ri, ci = linear_sum_assignment(cost)
    mp = dict(zip(ri, ci))
    zt = D["z"][te]
    zh = rho[te].argmax(1)
    sc = rho[te, mp[1]]
    g_hat = rho[te] @ np.array([gs[i] for i in gi])
    pear = pearson(g_hat, D["G_TRUE"][zt])

    def gain_pct(a, b):
        return float(100 * (a[te] - b[te]).mean() / a[te].mean())

    return dict(
        lam=lam, D1_g=g1, pi_true=D["pi"].tolist(), pi_hat=pi.tolist(),
        g_hat=[gs[i] for i in gi],
        ARI=float(ari(zh, zt)), AUC=float(auc(sc, zt)), pearson=float(pear),
        D4_gain_pct=gain_pct(d1, d4),
        MixTrue_gain_pct=gain_pct(d1, mt),
        D5_gain_pct=gain_pct(d1, d5),
        D0_gain_pct=gain_pct(d1, d0),
        D4_tost_ci=bootstrap_ci(d1[te] - d4[te]),
    )


def _fit_pi(ml, pi0=None, maxit=200, tol=1e-12):
    R = ml.shape[1]
    pi = np.full(R, 1.0 / R) if pi0 is None else np.array(pi0, float)
    for _ in range(maxit):
        lr = np.log(pi)[None] + ml
        rho = np.exp(lr - logsumexp(lr, 1, keepdims=True))
        pn = rho.mean(0); pn = pn / pn.sum()
        if np.abs(pn - pi).max() < tol:
            pi = pn
            break
        pi = pn
    return pi, logsumexp(np.log(pi)[None] + ml, 1).sum()


def _routing(ml_all, gi, pi):
    lr = np.log(pi)[None] + ml_all[:, list(gi)]
    return np.exp(lr - logsumexp(lr, 1, keepdims=True))


def dynamic_gate(runs):
    """Aggregate per-seed results into the design-gate + identifiability verdict.

    runs : list of dicts from `audit_dynamic`. Returns (design_gate, identifiability, summary).
    """
    def med(k):
        v = np.array([r[k] for r in runs], float)
        return (float(np.nanmedian(v)), float(np.nanpercentile(v, 25)),
                float(np.nanpercentile(v, 75)))

    ari_m, auc_m, pear_m = med("ARI"), med("AUC"), med("pearson")
    d5, mt, d0 = med("D5_gain_pct")[0], med("MixTrue_gain_pct")[0], med("D0_gain_pct")[0]
    design = bool(d5 >= 0.5 and mt >= 0.5 and d0 <= -0.5)
    ident = bool(ari_m[0] >= 0.6 and auc_m[0] >= 0.65 and pear_m[0] >= 0.6)
    summary = dict(
        n_seeds=len(runs),
        ARI=ari_m, AUC=auc_m, pearson=pear_m,
        D5_gain=d5, MixTrue_gain=mt, D0_gain=d0, D4_gain=med("D4_gain_pct")[0],
        design_gate=design, identifiability=ident,
    )
    return design, ident, summary
