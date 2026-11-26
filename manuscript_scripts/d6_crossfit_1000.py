# -*- coding: utf-8 -*-
"""1,000-draw 50/50 student cross-fitting + empirical theta-variance diagnostic."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import concurrent.futures as cf
import json
import numpy as np

import mechanism_common as mc
from d1_1000_perm_check import est_a

SIG = mc.SIG


def subset_by_users(d, mask_u):
    u_old = np.flatnonzero(mask_u)
    remap = np.full(d["n_u"], -1, dtype=np.int64)
    remap[u_old] = np.arange(len(u_old), dtype=np.int64)
    ev_mask = remap[d["qu"]] >= 0
    qu = remap[d["qu"][ev_mask]]
    qit = d["qit"][ev_mask]
    y = d["y"][ev_mask]
    kc_list = d["kc_list"]
    is_single = (d["K"] == 406)
    if is_single:
        owner = np.arange(len(y), dtype=np.int64)
        conc = np.array([kc_list[it][0] for it in qit], dtype=np.int64)
    else:
        lens = np.array([len(kc_list[it]) for it in qit], dtype=np.int64)
        owner = np.repeat(np.arange(len(y), dtype=np.int64), lens)
        conc = np.concatenate([np.asarray(kc_list[it], dtype=np.int64) - 1 for it in qit])
    return dict(
        name=d["name"], qu=qu, qit=qit, y=y, ev_owner=owner, ev_c=conc,
        K=d["K"], n_u=len(u_old), n_item=d["n_item"], kc_list=kc_list,
    )


def fit_theta_frozen_2pl(d, dv, a, iters=8):
    qu, qit, y = d["qu"], d["qit"], d["y"]
    th = np.zeros(d["n_u"], dtype=np.float64)
    a_q = a[qit]
    dv_q = dv[qit]
    for _ in range(iters):
        p = SIG(a_q * (th[qu] - dv_q))
        g = np.bincount(qu, weights=a_q * (y - p), minlength=d["n_u"])
        h = np.bincount(qu, weights=(a_q ** 2) * p * (1.0 - p), minlength=d["n_u"]) + 1.0
        th += g / h
    return th


def residual_a(d, theta, dv, a, min_n=1):
    res = d["y"] - SIG(a[d["qit"]] * (theta[d["qu"]] - dv[d["qit"]]))
    K, n_u = d["K"], d["n_u"]
    flat = d["qu"][d["ev_owner"]] * K + d["ev_c"]
    num = np.bincount(flat, weights=res[d["ev_owner"]], minlength=n_u * K)
    cnt = np.bincount(flat, minlength=n_u * K).astype(np.float64)
    return np.where(cnt >= min_n, num / np.maximum(cnt, 1), np.nan).reshape(n_u, K)


def _fast_top1_pow(R, n_iter=120):
    obs = ~np.isnan(R)
    mu = np.nanmean(R, axis=0)
    Rc = np.where(obs, R - mu, 0.0)
    v = np.random.default_rng(123).normal(size=Rc.shape[1])
    v /= np.linalg.norm(v)
    prev = 0.0
    for it in range(n_iter):
        u = Rc @ v
        nu = float(np.linalg.norm(u))
        if nu < 1e-12:
            return 0.0
        v = Rc.T @ u
        nv = float(np.linalg.norm(v))
        if nv < 1e-12:
            return 0.0
        v /= nv
        if abs(nu - prev) < 1e-6 and it >= 20:
            return nu
        prev = nu
    return float(np.linalg.norm(Rc @ v))


def _worker_draws(args):
    d_eval, dv_fit, a_fit, th_mean, th_std, n_draws, seed = args
    rng = np.random.default_rng(seed)
    tops = np.empty(n_draws, dtype=np.float64)
    qu, qit = d_eval["qu"], d_eval["qit"]
    n_u, K = d_eval["n_u"], d_eval["K"]
    ev_owner, ev_c = d_eval["ev_owner"], d_eval["ev_c"]
    a_q, dv_q = a_fit[qit], dv_fit[qit]
    flat = qu[ev_owner] * K + ev_c
    cnt = np.bincount(flat, minlength=n_u * K).astype(np.float64)
    mask_obs = cnt >= 1
    for b in range(n_draws):
        th_sim = rng.normal(th_mean, th_std, n_u)
        p_sim = SIG(a_q * (th_sim[qu] - dv_q))
        y_sim = (rng.random(len(p_sim)) < p_sim).astype(np.float64)
        th_hat = np.zeros(n_u, dtype=np.float64)
        for _ in range(6):
            p = SIG(a_q * (th_hat[qu] - dv_q))
            g = np.bincount(qu, weights=a_q * (y_sim - p), minlength=n_u)
            h = np.bincount(qu, weights=(a_q ** 2) * p * (1.0 - p), minlength=n_u) + 1.0
            th_hat += g / h
        res = y_sim - SIG(a_q * (th_hat[qu] - dv_q))
        num = np.bincount(flat, weights=res[ev_owner], minlength=n_u * K)
        R = np.where(mask_obs, num / np.maximum(cnt, 1.0), np.nan).reshape(n_u, K)
        tops[b] = _fast_top1_pow(R)
    return tops


def run_dataset_1000(d, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    perm_u = rng.permutation(d["n_u"])
    half = d["n_u"] // 2
    mask_fit = np.zeros(d["n_u"], dtype=bool)
    mask_fit[perm_u[:half]] = True
    d_fit = subset_by_users(d, mask_fit)
    d_eval = subset_by_users(d, ~mask_fit)

    th_fit, dv_fit = mc.rasch_fit(d_fit)
    a_fit = est_a(d_fit, th_fit, dv_fit, iters=20, ridge=20.0)
    th_eval = fit_theta_frozen_2pl(d_eval, dv_fit, a_fit, iters=8)
    real_top1 = _fast_top1_pow(residual_a(d_eval, th_eval, dv_fit, a_fit))

    emp_th_mean = float(np.mean(th_eval))
    emp_th_std = float(np.std(th_eval, ddof=1))

    with cf.ProcessPoolExecutor(max_workers=4) as ex:
        tops = np.concatenate(list(ex.map(
            _worker_draws, [(d_eval, dv_fit, a_fit, 0.0, 1.0, n_boot // 4, 1000 + w) for w in range(4)]
        )))
        tops_emp = np.concatenate(list(ex.map(
            _worker_draws, [(d_eval, dv_fit, a_fit, emp_th_mean, emp_th_std, 50, 2000 + w) for w in range(4)]
        )))

    th_full, dv_full = mc.rasch_fit(d)
    a_full = est_a(d, th_full, dv_full, iters=20, ridge=20.0)
    return {
        "dataset": d["name"],
        "n_fit": int(d_fit["n_u"]),
        "n_eval": int(d_eval["n_u"]),
        "crossfit_real_top1": real_top1,
        "anchor_n01_median": float(np.median(tops)),
        "anchor_n01_ci95": [float(np.percentile(tops, 2.5)), float(np.percentile(tops, 97.5))],
        "ratio_n01": float(real_top1 / np.median(tops)),
        "p_value_n01": float((1 + np.sum(tops >= real_top1)) / (1 + len(tops))),
        "emp_theta_std": emp_th_std,
        "anchor_emp_median": float(np.median(tops_emp)),
        "ratio_emp": float(real_top1 / np.median(tops_emp)),
        "corr_a_half_vs_full": float(np.corrcoef(a_fit, a_full)[0, 1]),
        "corr_d_half_vs_full": float(np.corrcoef(dv_fit, dv_full)[0, 1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=mc.default_data_root())
    parser.add_argument("--out", default="./results/crossfit_1000.json")
    args = parser.parse_args()
    os.environ["EDM_DATA_DIR"] = args.data_dir
    out = {"kdd_algebra": run_dataset_1000(mc.load_kdd()), "a0910": run_dataset_1000(mc.load_a0910())}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
