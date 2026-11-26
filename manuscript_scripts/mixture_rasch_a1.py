# -*- coding: utf-8 -*-
"""
Standalone Rost (1990) Dichotomous Mixture Rasch evaluation on 65/15/20 real splits
and synthetic difficulty-shift MDE sweep (zero external directory dependencies).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
import numpy as np
from sklearn.metrics import roc_auc_score

import data_splits as DS

_GH_X, _GH_W = np.polynomial.hermite.hermgauss(11)
THETA_Q = np.sqrt(2.0) * _GH_X
LOG_WQ = np.log(_GH_W / np.sqrt(np.pi))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _log_bern_student_grid(u: np.ndarray, it: np.ndarray, y: np.ndarray, n_u: int, d_r: np.ndarray) -> np.ndarray:
    Q = len(THETA_Q)
    logits = THETA_Q[None, :] - d_r[it, None]
    ll_ev = y[:, None] * logits - np.logaddexp(0.0, logits)
    flat_idx = (u[:, None] * Q + np.arange(Q, dtype=np.int64)[None, :]).ravel()
    return np.bincount(flat_idx, weights=ll_ev.ravel(), minlength=n_u * Q).reshape(n_u, Q)


def fit_rasch_1pl(u: np.ndarray, it: np.ndarray, y: np.ndarray, n_u: int, n_items: int, ridge_d: float = 1.0):
    cnt = np.bincount(it, minlength=n_items).astype(np.float64)
    pos = np.bincount(it, weights=y, minlength=n_items)
    p0 = (pos + 1.0) / (cnt + 2.0)
    d = -np.log(p0 / (1.0 - p0))
    d -= d.mean()
    for _ in range(12):
        ll_uq = _log_bern_student_grid(u, it, y, n_u, d) + LOG_WQ[None, :]
        m_u = ll_uq.max(axis=1, keepdims=True)
        post_uq = np.exp(ll_uq - m_u)
        post_uq /= post_uq.sum(axis=1, keepdims=True)
        w_eq = post_uq[u, :]
        p_eq = _sigmoid(THETA_Q[None, :] - d[it, None])
        g = np.bincount(it, weights=(-(y[:, None] - p_eq) * w_eq).sum(axis=1), minlength=n_items) - ridge_d * d
        h = np.bincount(it, weights=(p_eq * (1.0 - p_eq) * w_eq).sum(axis=1), minlength=n_items) + ridge_d
        d += np.clip(g / h, -1.0, 1.0)
    return d


def fit_mixture_rasch(u, it, y, n_u, n_items, R, d_init, n_restarts=2, n_iters=12, ridge_d=2.0, seed=0):
    if R == 1:
        return np.array([1.0]), d_init[None, :].copy()
    rng = np.random.default_rng(seed)
    best_ll, best_pi, best_d = -np.inf, None, None
    for _ in range(n_restarts):
        pi = np.full(R, 1.0 / R, dtype=np.float64)
        d_mat = np.tile(d_init, (R, 1)) + rng.normal(0.0, 0.35, size=(R, n_items))
        prev_ll = -np.inf
        for it_em in range(n_iters):
            log_pu_r = np.zeros((n_u, R), dtype=np.float64)
            post_urq = np.zeros((n_u, R, len(THETA_Q)), dtype=np.float64)
            for r in range(R):
                ll_uq = _log_bern_student_grid(u, it, y, n_u, d_mat[r]) + LOG_WQ[None, :]
                mx = ll_uq.max(axis=1, keepdims=True)
                ex = np.exp(ll_uq - mx)
                sm = ex.sum(axis=1, keepdims=True)
                log_pu_r[:, r] = mx[:, 0] + np.log(sm[:, 0])
                post_urq[:, r, :] = ex / sm
            log_joint = log_pu_r + np.log(np.maximum(pi, 1e-12))[None, :]
            mx_r = log_joint.max(axis=1, keepdims=True)
            ex_r = np.exp(log_joint - mx_r)
            sm_r = ex_r.sum(axis=1, keepdims=True)
            ll_tot = float((mx_r[:, 0] + np.log(sm_r[:, 0])).sum())
            gamma_ur = ex_r / sm_r
            pi = np.maximum(gamma_ur.mean(axis=0), 1e-4)
            pi /= pi.sum()
            for r in range(R):
                w_eq = gamma_ur[u, r][:, None] * post_urq[u, r, :]
                p_eq = _sigmoid(THETA_Q[None, :] - d_mat[r, it, None])
                g = np.bincount(it, weights=(-(y[:, None] - p_eq) * w_eq).sum(axis=1), minlength=n_items) - ridge_d * (d_mat[r] - d_init)
                h = np.bincount(it, weights=(p_eq * (1.0 - p_eq) * w_eq).sum(axis=1), minlength=n_items) + ridge_d
                d_mat[r] += np.clip(g / h, -1.0, 1.0)
            if abs(ll_tot - prev_ll) < 1e-2 and it_em >= 5:
                break
            prev_ll = ll_tot
        if prev_ll > best_ll:
            best_ll, best_pi, best_d = prev_ll, pi.copy(), d_mat.copy()
    return best_pi, best_d


def eval_support_conditioned(su, sit, sy, qu, qit, qy, n_u, pi, d_mat):
    R = len(pi)
    log_s_r = np.zeros((n_u, R), dtype=np.float64)
    post_s_urq = np.zeros((n_u, R, len(THETA_Q)), dtype=np.float64)
    for r in range(R):
        ll_uq = _log_bern_student_grid(su, sit, sy, n_u, d_mat[r]) + LOG_WQ[None, :]
        mx = ll_uq.max(axis=1, keepdims=True)
        ex = np.exp(ll_uq - mx)
        sm = ex.sum(axis=1, keepdims=True)
        log_s_r[:, r] = mx[:, 0] + np.log(sm[:, 0])
        post_s_urq[:, r, :] = ex / sm
    log_joint = log_s_r + np.log(np.maximum(pi, 1e-12))[None, :]
    mx_r = log_joint.max(axis=1, keepdims=True)
    ex_r = np.exp(log_joint - mx_r)
    rho = ex_r / ex_r.sum(axis=1, keepdims=True)
    p_pred = np.zeros(len(qu), dtype=np.float64)
    for r in range(R):
        p_qr = _sigmoid(THETA_Q[None, :] - d_mat[r, qit, None])
        p_pred += rho[qu, r] * (post_s_urq[qu, r, :] * p_qr).sum(axis=1)
    p_pred = np.clip(p_pred, 1e-7, 1.0 - 1e-7)
    nll_ev = -(qy * np.log(p_pred) + (1.0 - qy) * np.log(1.0 - p_pred))
    return float(nll_ev.mean()), float(roc_auc_score(qy, p_pred)), rho


def build_common_eval_arrays(df, n_u, M, maxM, split_seed, max_q=0):
    rng = np.random.default_rng(split_seed)
    grp = {int(u): g for u, g in df.groupby("u_idx")}
    su, sit, sy, qu, qit, qy = [], [], [], [], [], []
    for u in range(n_u):
        g = grp[u]
        its, sc = g["item_idx"].values.astype(np.int64), g["score"].values.astype(np.float64)
        perm = rng.permutation(len(its))
        s_idx, q_idx = perm[:M], perm[maxM:]
        if max_q > 0 and len(q_idx) > max_q:
            q_idx = q_idx[:max_q]
        su.append(np.full(len(s_idx), u, dtype=np.int64)); sit.append(its[s_idx]); sy.append(sc[s_idx])
        qu.append(np.full(len(q_idx), u, dtype=np.int64)); qit.append(its[q_idx]); qy.append(sc[q_idx])
    return tuple(np.concatenate(x) for x in (su, sit, sy, qu, qit, qy))


def run_synthetic_mde():
    out = {}
    for sig_delta in (0.0, 0.75, 1.25, 1.75):
        gains, aucs = [], []
        for seed in (101, 102, 103, 104):
            rng = np.random.default_rng(seed + int(sig_delta * 100))
            n_u, n_items, n_ev = 1200, 220, 60
            d_base = rng.normal(0.0, 1.0, n_items)
            shift = rng.normal(0.0, sig_delta, n_items)
            d_true = np.vstack([d_base - 0.5 * shift, d_base + 0.5 * shift])
            z_true = rng.integers(0, 2, size=n_u)
            th_true = rng.normal(0.0, 1.0, n_u)
            u_all, it_all, y_all = [], [], []
            for i in range(n_u):
                its = rng.choice(n_items, size=n_ev, replace=False)
                ys = (rng.random(n_ev) < _sigmoid(th_true[i] - d_true[z_true[i], its])).astype(np.float64)
                u_all.append(np.full(n_ev, i, dtype=np.int64)); it_all.append(its); y_all.append(ys)
            n_tr, n_tu = int(0.65 * n_u), int(0.15 * n_u)
            d_init = fit_rasch_1pl(np.concatenate(u_all[:n_tr]), np.concatenate(it_all[:n_tr]), np.concatenate(y_all[:n_tr]), n_tr, n_items, 1.0)
            pi2, d2 = fit_mixture_rasch(np.concatenate(u_all[:n_tr]), np.concatenate(it_all[:n_tr]), np.concatenate(y_all[:n_tr]), n_tr, n_items, 2, d_init, 2, 12, 2.0, seed)
            te_it_arr, te_y_arr = it_all[n_tr + n_tu:], y_all[n_tr + n_tu:]
            n_te = len(te_it_arr)
            su = np.concatenate([np.full(20, i, dtype=np.int64) for i in range(n_te)])
            sit = np.concatenate([te_it_arr[i][:20] for i in range(n_te)])
            sy = np.concatenate([te_y_arr[i][:20] for i in range(n_te)])
            qu = np.concatenate([np.full(40, i, dtype=np.int64) for i in range(n_te)])
            qit = np.concatenate([te_it_arr[i][20:] for i in range(n_te)])
            qy = np.concatenate([te_y_arr[i][20:] for i in range(n_te)])
            nll1, _, _ = eval_support_conditioned(su, sit, sy, qu, qit, qy, n_te, np.array([1.0]), d_init[None, :])
            nll2, _, rho = eval_support_conditioned(su, sit, sy, qu, qit, qy, n_te, pi2, d2)
            rauc = float(roc_auc_score(z_true[n_tr + n_tu:], rho[:, 1]))
            gains.append(100.0 * (nll1 - nll2) / nll1)
            aucs.append(max(rauc, 1.0 - rauc))
        out[str(sig_delta)] = {"gain_r2_med": float(np.median(gains)), "routing_auc_med": float(np.median(aucs))}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DS.default_data_dir())
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--out", default="./results/mixture_rasch.json")
    args = parser.parse_args()

    res = {"synthetic_mde": run_synthetic_mde()}
    if not args.synthetic_only:
        real = {}
        for ds_name, loader, max_q in (("a0910", DS.load_a0910_3way, 0), ("kdd_algebra", DS.load_kdd_algebra_3way, 200)):
            seeds_out = []
            for seed in (42, 43, 44):
                tr, _, te_df, _, n_tr, _, n_te, n_items, _ = loader(seed=seed, data_dir=args.data_dir)
                tr_u, tr_it, tr_y = tr[0], tr[1], tr[2].astype(np.float64)
                d_init = fit_rasch_1pl(tr_u, tr_it, tr_y, n_tr, n_items, 1.0)
                pi2, d2 = fit_mixture_rasch(tr_u, tr_it, tr_y, n_tr, n_items, 2, d_init, 2, 12, 2.0, seed + 2)
                m_out = {}
                for M in (10, 20):
                    su, sit, sy, qu, qit, qy = build_common_eval_arrays(te_df, n_te, M, 20, seed + 1000, max_q=max_q)
                    nll1, auc1, _ = eval_support_conditioned(su, sit, sy, qu, qit, qy, n_te, np.array([1.0]), d_init[None, :])
                    nll2, auc2, _ = eval_support_conditioned(su, sit, sy, qu, qit, qy, n_te, pi2, d2)
                    m_out[str(M)] = {"r1_nll": nll1, "r2_nll": nll2, "gain_pct": 100.0 * (nll1 - nll2) / nll1, "r1_auc": auc1, "r2_auc": auc2}
                seeds_out.append({"seed": seed, "M": m_out})
            real[ds_name] = seeds_out
        res["real"] = real
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
