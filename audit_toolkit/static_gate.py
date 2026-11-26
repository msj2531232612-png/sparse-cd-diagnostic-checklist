"""Static mode-mixture screen (Rasch readout).

Compares a scalar ridge (B0) against a mixture of per-mode shrinkage strengths
(B2) with a per-student oracle ceiling (B4_qopt). The manuscript's main table
uses a frozen HierReadout; this module is the Rasch reference implementation
used by the positive control, kept here so the gate logic is self-contained.
"""
import itertools
import numpy as np
from scipy.special import expit, logsumexp

from .core_laplace import logmarg_rasch, fit_mixture_pi, softmax_resp, ari, auc

LAMS_DEFAULT = np.array([0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4])


def generate_static(seed, N=600, J=220, n_per=60, var_true=(1.0, 1.0), pi=(0.5, 0.5)):
    """Generate Rasch responses with optional two-mode ability-variance structure.

    var_true : per-mode ability variance. Equal variances give the null (no mode
               structure); strongly distinct variances give a detectable signal.
    """
    rng = np.random.default_rng(seed)
    d = rng.normal(0, 1, J)
    p = np.asarray(pi, float); p = p / p.sum()
    z = rng.choice(len(var_true), N, p=p)
    th = rng.normal(0, np.sqrt(np.asarray(var_true, float)[z]))
    items = np.stack([rng.choice(J, n_per, replace=False) for _ in range(N)])
    y = (rng.random((N, n_per)) < expit(th[:, None] - d[items])).astype(float)
    return dict(d=d, z=z, items=items, y=y)


def audit_static(d, items, y, n_fit=40, lams=LAMS_DEFAULT, Q=48, z=None, n_oracle=1, mask=None):
    """Run B0 (single global lambda) and B2 (mixture EM) on the first `n_fit`
    responses, evaluate the mixture on the held-out remainder.

    Returns a dict with keys:
      lam_single, lam_mix, pi_hat, heldout_gain_pct, bic_picks_R2,
      route_auc, route_ari, qopt_gain_pct, n_eval_users
    """
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        fit_m, ho_m = mask[:, :n_fit], mask[:, n_fit:]
        valid = (fit_m.sum(1) > 0) & (ho_m.sum(1) > 0)
        if not np.any(valid):
            raise ValueError("No evaluation users have both non-empty support (fit) and query (held-out) responses.")
        items, y = items[valid], y[valid]
        fit_m, ho_m = fit_m[valid], ho_m[valid]
        if z is not None:
            z = np.asarray(z)[valid]
    else:
        fit_m, ho_m = None, None

    fit_i, fit_y = items[:, :n_fit], y[:, :n_fit]
    ho_i, ho_y = items[:, n_fit:], y[:, n_fit:]
    if fit_i.shape[1] == 0 or ho_i.shape[1] == 0:
        raise ValueError("Both support (fit) and query (held-out) portions must be non-empty.")

    lm = np.stack([logmarg_rasch(d, fit_i, fit_y, lam, Q, mask=fit_m) for lam in lams], 1)  # (N, R)

    # B0: one shared lambda (chosen on total fit log-marginal, NOT per-student)
    best_l = int(lm.sum(0).argmax())
    lam_single = float(lams[best_l])
    single_ll = lm[:, best_l].sum()

    # B2: mixture EM over every lambda pair
    best = None
    for a, b in itertools.combinations(range(len(lams)), 2):
        pi, llv = fit_mixture_pi(lm[:, [a, b]])
        if best is None or llv > best[0]:
            best = (llv, (a, b), pi)
    mix_ll, (ia, ib), pi_hat = best

    # held-out unconditional marginal NLL (kept for positive-control compatibility)
    ho_single = logmarg_rasch(d, ho_i, ho_y, lam_single, Q, mask=ho_m)
    ho_comp = np.stack([logmarg_rasch(d, ho_i, ho_y, lams[ia], Q, mask=ho_m),
                        logmarg_rasch(d, ho_i, ho_y, lams[ib], Q, mask=ho_m)], 1)
    ho_mix = logsumexp(np.log(pi_hat)[None] + ho_comp, 1)
    gain_pct = float(100.0 * np.median((ho_mix - ho_single) / np.abs(ho_single)))

    # routing from the fit-portion soft assignment (exact log-posterior)
    lr = np.log(pi_hat)[None, :] + lm[:, [ia, ib]]
    log_rho = lr - logsumexp(lr, 1, keepdims=True)
    rho = np.exp(log_rho)
    raw_auc = auc(rho[:, 1], z) if z is not None else float("nan")
    route_auc = max(raw_auc, 1.0 - raw_auc) if z is not None else float("nan")
    route_ari = ari(rho.argmax(1), z) if z is not None else float("nan")

    # support-conditioned joint query log-marginal: log p(Q | S) = log p(S, Q) - log p(S)
    all_i = np.concatenate([fit_i, ho_i], axis=1)
    all_y = np.concatenate([fit_y, ho_y], axis=1)
    all_m = np.concatenate([fit_m, ho_m], axis=1) if (fit_m is not None and ho_m is not None) else None
    n_q = ho_m.sum(1).astype(float) if ho_m is not None else np.full(ho_i.shape[0], float(ho_i.shape[1]))

    joint_single = logmarg_rasch(d, all_i, all_y, lam_single, Q, mask=all_m)
    cond_single = joint_single - lm[:, best_l]
    joint_comp = np.stack([logmarg_rasch(d, all_i, all_y, lams[ia], Q, mask=all_m),
                           logmarg_rasch(d, all_i, all_y, lams[ib], Q, mask=all_m)], 1)
    cond_comp = joint_comp - lm[:, [ia, ib]]
    cond_mix = logsumexp(log_rho + cond_comp, 1)

    nll_single_stu = -cond_single / n_q
    nll_mix_stu = -cond_mix / n_q
    mean_single = float(nll_single_stu.mean())
    mean_mix = float(nll_mix_stu.mean())
    cond_gain_pct = float(100.0 * (mean_single - mean_mix) / max(abs(mean_single), 1e-12))

    # paired student-level bootstrap 95% CI (equal student weight)
    rng_boot = np.random.default_rng(0)
    n_stu = len(nll_single_stu)
    boot_gains = np.empty(500, dtype=float)
    for b_idx in range(500):
        idx = rng_boot.integers(0, n_stu, size=n_stu)
        ms = nll_single_stu[idx].mean()
        mm = nll_mix_stu[idx].mean()
        boot_gains[b_idx] = 100.0 * (ms - mm) / max(abs(ms), 1e-12)
    cond_gain_ci95 = [float(np.percentile(boot_gains, 2.5)), float(np.percentile(boot_gains, 97.5))]

    # post-hoc hard selection over the fixed candidate set (query-informed diagnostic)
    n_ho = ho_i.shape[1]
    qopt_gain = float("nan")
    if n_oracle > 1 and n_ho >= n_oracle:
        ho_all = np.stack([logmarg_rasch(d, ho_i, ho_y, lam, Q, mask=ho_m) for lam in lams], 1)
        best_per_stu = ho_all.max(1)
        qopt_gain = float(100.0 * np.median((best_per_stu - ho_single) / np.abs(ho_single)))

    bic_s = 2 * single_ll - 1 * np.log(fit_i.shape[0])
    bic_m = 2 * mix_ll - 3 * np.log(fit_i.shape[0])

    return dict(
        lam_single=lam_single,
        lam_mix=[float(lams[ia]), float(lams[ib])],
        pi_hat=pi_hat.tolist(),
        heldout_gain_pct=gain_pct,
        cond_nll_single=mean_single,
        cond_nll_mix=mean_mix,
        cond_gain_pct=cond_gain_pct,
        cond_gain_ci95=cond_gain_ci95,
        bic_picks_R2=bool(bic_m > bic_s),
        route_auc=route_auc,
        route_ari=route_ari,
        qopt_gain_pct=qopt_gain,
        n_eval_users=int(fit_i.shape[0]),
    )


def classify_static_verdict(res, nll_threshold=1.0, min_eval_users=20):
    """Map a real-data static audit dict to the three-state decision."""
    if res.get("n_eval_users", 0) < min_eval_users:
        return "insufficient evidence"
    gain = res.get("cond_gain_pct")
    ci = res.get("cond_gain_ci95")
    if gain is None or not np.isfinite(gain) or ci is None or len(ci) != 2:
        return "insufficient evidence"
    if not (np.isfinite(ci[0]) and np.isfinite(ci[1])) or ci[0] > ci[1]:
        return "insufficient evidence"
    if gain >= nll_threshold and ci[0] > 0.0 and res.get("bic_picks_R2", False):
        return "proceed"
    return "no improvement in tested configuration"


def static_gate(results):
    """Evaluates static mixture screen criteria (1% MID + routing AUC >= 0.65). `results` is a list of
    per-seed dicts from `audit_static`. Returns (verdict, summary)."""
    gains = np.array([r["heldout_gain_pct"] for r in results])
    aucs = np.array([r["route_auc"] for r in results])
    bic2 = sum(1 for r in results if r["bic_picks_R2"])
    g0_1 = bool(np.median(gains) >= 1.0)
    g0_2 = bool(np.median(aucs) >= 0.65)
    summary = dict(
        n_seeds=len(results),
        gain_median=float(np.median(gains)),
        gain_q25=float(np.percentile(gains, 25)),
        gain_q75=float(np.percentile(gains, 75)),
        bic_picks_R2=f"{bic2}/{len(results)}",
        route_auc_median=float(np.median(aucs)),
        g0_1=g0_1, g0_2=g0_2,
    )
    verdict = "RELEASE" if (g0_1 and g0_2) else "REJECT"
    return verdict, summary
