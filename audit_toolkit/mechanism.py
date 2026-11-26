"""D1 residual-spectrum diagnostic (mechanism analysis).

Fits a Rasch readout, builds the student x concept residual matrix, and compares
its top singular values against a per-column permutation null and a homogeneous
power anchor. This is the manuscript's M3 test for exploitable low-rank mode
structure beyond item-level heteroscedasticity.
"""
import numpy as np
from scipy.special import expit


def fit_rasch(items, y, J, lam_theta=1.0, n_iter=6, mask=None):
    """Alternating Newton: item difficulties d_j (ML) and student theta_i (MAP).

    Returns (d (J,), theta (N,)).
    """
    N, _n = items.shape
    obs_mask = np.ones_like(items, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    obs_f = obs_mask.astype(float)
    d = np.zeros(J)
    theta = np.zeros(N)
    for _ in range(n_iter):
        # theta step (MAP with N(0, 1/lam_theta) prior)
        p = expit(theta[:, None] - d[items])
        g = ((y - p) * obs_f).sum(1) - lam_theta * theta
        h = -p * (1 - p) * obs_f
        H = h.sum(1) - lam_theta
        theta = theta - g / np.where(H != 0, H, -lam_theta)
        # d step (ML)
        p_cur = expit(theta[:, None] - d[items])
        for j in range(J):
            m = (items == j) & obs_mask
            if not np.any(m):
                continue
            pj = p_cur[m]
            yj = y[m]
            g_j = (pj - yj).sum()
            h_j = (pj * (1 - pj)).sum()
            if h_j > 0:
                d[j] = d[j] - g_j / h_j
    return d, theta


def fit_theta_frozen(items, y, d, lam_theta=1.0, n_iter=6, mask=None):
    """Fit student ability theta_i (MAP) with item difficulties d frozen."""
    N, _n = items.shape
    obs_f = np.ones_like(items, dtype=float) if mask is None else np.asarray(mask, dtype=float)
    theta = np.zeros(N)
    for _ in range(n_iter):
        p = expit(theta[:, None] - d[items])
        g = ((y - p) * obs_f).sum(1) - lam_theta * theta
        h = -p * (1 - p) * obs_f
        H = h.sum(1) - lam_theta
        theta = theta - g / np.where(H != 0, H, -lam_theta)
    return theta


def residual_matrix(items, y, d, theta, concept_of_item, K, mask=None):
    """student x concept residual r_{i,c} = mean_i(y - p) over responses to c.

    `concept_of_item` is a 1-D item -> concept map (indexed by item id).
    """
    p = expit(theta[:, None] - d[items])
    r = y - p
    N = items.shape[0]
    R = np.zeros((N, K))
    counts = np.zeros((N, K))
    for i in range(N):
        for pos in range(items.shape[1]):
            if mask is not None and not mask[i, pos]:
                continue
            c = int(concept_of_item[items[i, pos]])
            R[i, c] += r[i, pos]
            counts[i, c] += 1
    R = np.where(counts > 0, R / np.maximum(counts, 1), np.nan)
    return R


def spectrum(R, n_perm=200, seed=0, top=5):
    """Column-centered SVD top-k singular values + per-column permutation null.

    Returns dict(topk=real singular values, null95=95th percentile under the
    permutation null, null_median=median under null).
    """
    Rc = np.where(np.isnan(R), 0.0, R - np.nanmean(R, 0, keepdims=True))
    _, s, _ = np.linalg.svd(Rc, full_matrices=False)
    topk = s[:top]

    rng = np.random.default_rng(seed)
    null_s = []
    for _ in range(n_perm):
        Rp = Rc.copy()
        for c in range(Rc.shape[1]):
            Rp[:, c] = rng.permutation(Rp[:, c])
        _, sp, _ = np.linalg.svd(Rp, full_matrices=False)
        null_s.append(sp[:top])
    null_s = np.array(null_s)
    return dict(topk=topk.tolist(),
                null95=np.percentile(null_s, 95, axis=0).tolist(),
                null_median=np.median(null_s, axis=0).tolist())


def homogeneous_anchor(d, items, concept_of_item, K, n_sim=20, seed=0, top=1, mask=None, frozen_d=False):
    """Power anchor: run the SAME fit-and-SVD pipeline on data generated from a
    homogeneous model (theta ~ N(0,1), same items/d), so the anchor shares the
    estimator's noise floor. Returns the median top singular value over n_sim.
    """
    rng = np.random.default_rng(seed)
    N, n = items.shape
    vals = []
    for _ in range(n_sim):
        theta = rng.normal(0, 1, N)
        y = (rng.random((N, n)) < expit(theta[:, None] - d[items])).astype(float)
        if frozen_d:
            d_hat = d
            th_hat = fit_theta_frozen(items, y, d, mask=mask)
        else:
            d_hat, th_hat = fit_rasch(items, y, d.size, mask=mask)
        R = residual_matrix(items, y, d_hat, th_hat, concept_of_item, K, mask=mask)
        Rc = np.where(np.isnan(R), 0.0, R - np.nanmean(R, 0, keepdims=True))
        _, s, _ = np.linalg.svd(Rc, full_matrices=False)
        vals.append(s[0] if top == 1 else s[:top][-1])
    return float(np.median(vals))
