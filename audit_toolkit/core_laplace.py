"""Gauss--Hermite quadrature, Rasch log-marginals and metric utilities.

NumPy + SciPy only. Everything is seedable and takes data as arguments so the
toolkit never touches the filesystem for data paths (see `run_audit.py` for I/O).
"""
import numpy as np
from scipy.special import expit, logsumexp
from scipy.stats import rankdata, pearsonr

_GH_CACHE = {}


def gh_nodes(lam, Q=48):
    """Gauss--Hermite nodes/weights for an ability prior N(0, 1/lam).

    Returns (theta (Q,), logw (Q,)) with the nodes rescaled by sqrt(2/lam).
    """
    key = (float(lam), int(Q))
    if key not in _GH_CACHE:
        x, w = np.polynomial.hermite.hermgauss(Q)
        theta = np.sqrt(2.0 / lam) * x
        logw = np.log(w / np.sqrt(np.pi))
        _GH_CACHE[key] = (theta, logw)
    return _GH_CACHE[key]


def logmarg_rasch(d, items, y, lam, Q=48, mask=None):
    """Per-student log-marginal likelihood of a Rasch (1-PL) model.

    d     : (J,) item difficulties
    items : (N, n) item index per response
    y     : (N, n) binary responses
    lam   : prior precision of theta ~ N(0, 1/lam)
    mask  : optional (N, n) boolean mask for valid (non-padded) entries
    Returns (N,) log-marginals.
    """
    theta, logw = gh_nodes(lam, Q)
    d_i = d[items]                                    # (N, n)
    eta = theta[None, :, None] - d_i[:, None, :]      # (N, Q, n)
    s = (2.0 * y - 1.0)[:, None, :]                   # (N, 1, n)
    ll = -np.logaddexp(0.0, -s * eta)                 # (N, Q, n)
    if mask is not None:
        ll = ll * np.asarray(mask, float)[:, None, :]
    return logsumexp(logw[None, :] + ll.sum(2), axis=1)


def softmax_resp(logpi, logmarg):
    """Posterior assignment probabilities given log priors and per-mode marginals."""
    lr = logpi[None, :] + logmarg
    return np.exp(lr - logsumexp(lr, 1, keepdims=True))


def fit_mixture_pi(logmarg, pi0=None, maxit=200, tol=1e-12):
    """EM for the mixture weights pi given (N, R) log-marginals.

    Returns (pi (R,), total_mixture_loglik (float)).
    """
    R = logmarg.shape[1]
    pi = np.full(R, 1.0 / R) if pi0 is None else np.array(pi0, float)
    for _ in range(maxit):
        lr = np.log(pi)[None, :] + logmarg
        rho = np.exp(lr - logsumexp(lr, 1, keepdims=True))
        pi_new = rho.mean(0)
        pi_new = pi_new / pi_new.sum()
        if np.abs(pi_new - pi).max() < tol:
            pi = pi_new
            break
        pi = pi_new
    return pi, logsumexp(np.log(pi)[None, :] + logmarg, 1).sum()


def ari(x, y):
    """Adjusted Rand index between two clusterings (labels as ints)."""
    x = np.asarray(x, int); y = np.asarray(y, int)
    c = np.zeros((x.max() + 1, y.max() + 1))
    for i, j in zip(x, y):
        c[i, j] += 1
    choose2 = lambda v: v * (v - 1) / 2
    s = choose2(c).sum()
    sa = choose2(c.sum(1)).sum()
    sb = choose2(c.sum(0)).sum()
    n2 = choose2(len(x))
    exp_ = sa * sb / n2
    mx = (sa + sb) / 2
    return 1.0 if mx == exp_ else (s - exp_) / (mx - exp_)


def auc(score, lab):
    """Area under the ROC curve; returns NaN if a class is empty."""
    if lab.min() == lab.max():
        return float("nan")
    r = rankdata(score)
    n1 = lab.sum(); n0 = len(lab) - n1
    return (r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(diff, B=2000, seed=7, lo=2.5, hi=97.5):
    """Paired bootstrap CI on the mean of `diff`."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    m = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(B)])
    return float(np.percentile(m, lo)), float(np.percentile(m, hi))


def pearson(a, b):
    """Pearson r, guarded against zero variance."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(pearsonr(a, b)[0])
