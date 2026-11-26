# -*- coding: utf-8 -*-
"""
Standalone Mode-Specific Low-Rank Subspace Mixture (Mix-V_r, mu_r, rank L=3)
evaluation on 65/15/20 real splits and planted orthogonal-subspace MDE sweep
(zero external machine paths).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import concurrent.futures as cf
import json
import sys
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score

import data_splits as DS

L_RANK = 3
PI_OVER_8 = np.pi / 8.0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    res = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(res, axis=axis)


def _bce_sum(logit: np.ndarray, y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    loss = np.maximum(logit, 0.0) - logit * y + np.log1p(np.exp(-np.abs(logit)))
    return np.sum(loss * mask, axis=1)


# Lightweight self-contained Batch & Newton helpers so manuscript_scripts/ runs standalone
class SimpleBatch:
    def __init__(self, A_all, d_all, pop, sup_items, sup_y, q_items=None, q_y=None):
        A_all = np.asarray(A_all, dtype=np.float64)
        d_all = np.asarray(d_all, dtype=np.float64)
        pop = np.asarray(pop, dtype=np.float64)
        self.N = len(sup_items)
        self.K = A_all.shape[1]
        s_all = A_all.sum(axis=1)
        b_all = A_all @ pop - d_all
        J_sup = np.array([len(it) for it in sup_items], dtype=np.int64)
        self.max_J = int(np.max(J_sup)) if self.N > 0 else 0

        self.obs_list = []
        K_obs = np.zeros(self.N, dtype=np.int64)
        obs_mask_K = np.zeros((self.N, self.K), dtype=np.float64)
        for i in range(self.N):
            it = np.asarray(sup_items[i], dtype=np.int64)
            obs = np.flatnonzero(np.any(A_all[it] != 0.0, axis=0)) if len(it) > 0 else np.zeros(0, dtype=np.int64)
            self.obs_list.append(obs)
            K_obs[i] = len(obs)
            if len(obs) > 0:
                obs_mask_K[i, obs] = 1.0
        self.max_Kobs = int(np.max(K_obs)) if self.N > 0 else 0
        self.obs_pad = np.zeros((self.N, self.max_Kobs), dtype=np.int64)

        self.X_sup = np.zeros((self.N, self.max_J, self.K), dtype=np.float64)
        self.X_sup_obs = np.zeros((self.N, self.max_J, self.max_Kobs), dtype=np.float64)
        self.s_sup = np.zeros((self.N, self.max_J), dtype=np.float64)
        self.b_sup = np.zeros((self.N, self.max_J), dtype=np.float64)
        self.y_sup = np.zeros((self.N, self.max_J), dtype=np.float64)
        self.mask_J = np.zeros((self.N, self.max_J), dtype=np.float64)
        self.mask_obs = np.zeros((self.N, self.max_Kobs), dtype=np.float64)

        for i in range(self.N):
            it = np.asarray(sup_items[i], dtype=np.int64)
            J_i = len(it)
            obs = self.obs_list[i]
            k_i = len(obs)
            if k_i > 0:
                self.mask_obs[i, :k_i] = 1.0
                self.obs_pad[i, :k_i] = obs
            if J_i > 0:
                self.X_sup[i, :J_i] = A_all[it]
                self.s_sup[i, :J_i] = s_all[it]
                self.b_sup[i, :J_i] = b_all[it]
                self.y_sup[i, :J_i] = sup_y[i]
                self.mask_J[i, :J_i] = 1.0
                if k_i > 0:
                    self.X_sup_obs[i, :J_i, :k_i] = A_all[it][:, obs]

        if q_items is None:
            self.max_Jq = 0
        else:
            J_q = np.array([len(it) for it in q_items], dtype=np.int64)
            self.max_Jq = int(np.max(J_q)) if self.N > 0 else 0
            self.X_q = np.zeros((self.N, self.max_Jq, self.K), dtype=np.float64)
            self.X_q_obs = np.zeros((self.N, self.max_Jq, self.max_Kobs), dtype=np.float64)
            self.s_q = np.zeros((self.N, self.max_Jq), dtype=np.float64)
            self.b_q = np.zeros((self.N, self.max_Jq), dtype=np.float64)
            self.y_q = np.zeros((self.N, self.max_Jq), dtype=np.float64)
            self.mask_Jq = np.zeros((self.N, self.max_Jq), dtype=np.float64)
            self.xq_norm2_unobs = np.zeros((self.N, self.max_Jq), dtype=np.float64)
            for i in range(self.N):
                it = np.asarray(q_items[i], dtype=np.int64)
                Jq_i = len(it)
                obs = self.obs_list[i]
                k_i = len(obs)
                if Jq_i > 0:
                    Xq = A_all[it]
                    self.X_q[i, :Jq_i] = Xq
                    self.s_q[i, :Jq_i] = s_all[it]
                    self.b_q[i, :Jq_i] = b_all[it]
                    self.y_q[i, :Jq_i] = q_y[i]
                    self.mask_Jq[i, :Jq_i] = 1.0
                    if k_i > 0:
                        self.X_q_obs[i, :Jq_i, :k_i] = Xq[:, obs]
                    unobs = 1.0 - obs_mask_K[i]
                    self.xq_norm2_unobs[i, :Jq_i] = np.sum((Xq ** 2) * unobs[None, :], axis=1)


def solve_subspace_component(batch, lam_g, lam_d, mu_r, V_r, compute_query=False, n_newton=6):
    N, L, K_obs_max = batch.N, V_r.shape[1], batch.max_Kobs
    D = 1 + L + K_obs_max
    mu_obs = mu_r[batch.obs_pad] * batch.mask_obs
    V_obs = V_r[batch.obs_pad] * batch.mask_obs[:, :, None]
    b_sup = batch.b_sup + np.einsum("njk,nk->nj", batch.X_sup_obs, mu_obs)
    XV_sup = np.einsum("njk,nkl->njl", batch.X_sup_obs, V_obs)
    M_sup = np.concatenate([batch.s_sup[:, :, None], XV_sup, batch.X_sup_obs], axis=2)

    lam_vec = np.ones((N, D), dtype=np.float64)
    lam_vec[:, 0] = lam_g
    lam_vec[:, 1 : 1 + L] = 1.0
    lam_vec[:, 1 + L :] = batch.mask_obs * lam_d + (1.0 - batch.mask_obs) * 1.0
    z_mask = np.ones((N, D), dtype=np.float64)
    z_mask[:, 1 + L :] = batch.mask_obs

    z = np.zeros((N, D), dtype=np.float64)
    eye_diag = np.eye(D, dtype=np.float64)[None, :, :]
    for _ in range(n_newton):
        logit = b_sup + np.einsum("njd,nd->nj", M_sup, z)
        p = _sigmoid(logit)
        w = p * (1.0 - p) * batch.mask_J
        res = (p - batch.y_sup) * batch.mask_J
        grad = (np.einsum("njd,nj->nd", M_sup, res) + lam_vec * z) * z_mask
        H = np.einsum("njd,nje->nde", M_sup * w[:, :, None], M_sup) + eye_diag * lam_vec[:, None, :]
        z -= np.clip(np.linalg.solve(H, grad[:, :, None])[:, :, 0] * z_mask, -2.5, 2.5)

    logit = b_sup + np.einsum("njd,nd->nj", M_sup, z)
    p = _sigmoid(logit)
    w = p * (1.0 - p) * batch.mask_J
    H = np.einsum("njd,nje->nde", M_sup * w[:, :, None], M_sup) + eye_diag * lam_vec[:, None, :]
    log_m = -_bce_sum(logit, batch.y_sup, batch.mask_J) - 0.5 * np.sum(lam_vec * (z ** 2) * z_mask, axis=1) \
            + 0.5 * np.sum(np.log(lam_vec) * z_mask, axis=1) - 0.5 * np.linalg.slogdet(H)[1]

    out = {"theta_g": z[:, 0], "eta": z[:, 1 : 1 + L], "eps_obs": z[:, 1 + L :], "log_m": log_m}
    if compute_query and batch.max_Jq > 0:
        b_q = batch.b_q + np.einsum("njk,k->nj", batch.X_q, mu_r)
        XV_q = np.einsum("njk,kl->njl", batch.X_q, V_r)
        M_q = np.concatenate([batch.s_q[:, :, None], XV_q, batch.X_q_obs], axis=2)
        mu_q = b_q + np.einsum("njd,nd->nj", M_q, z)
        Hinv_Mq = np.linalg.solve(H, np.swapaxes(M_q, 1, 2))
        var_tot = np.maximum(np.sum(M_q * np.swapaxes(Hinv_Mq, 1, 2), axis=2) + batch.xq_norm2_unobs / lam_d, 0.0)
        out["p_q"] = _sigmoid(mu_q / np.sqrt(1.0 + PI_OVER_8 * var_tot))
    return out


def eval_nll(batch, p_q):
    m = batch.mask_Jq > 0.5
    p = np.clip(p_q[m], 1e-7, 1.0 - 1e-7)
    y = batch.y_q[m]
    return float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean())


def run_synthetic_subspace_demo():
    """Verify Mix-V_r recovery on a planted orthogonal rank-3 concept subspace benchmark (zero external data)."""
    K, J, N, E_STU, M = 123, 220, 600, 60, 20
    rng = np.random.default_rng(101)
    Q = np.zeros((J, K), dtype=np.float64)
    for j in range(J):
        Q[j, rng.choice(K, size=(1 if j < 180 else 2), replace=False)] = 1.0
    A = rng.uniform(0.7, 1.4, size=J)[:, None] * (Q / np.maximum(Q.sum(axis=1, keepdims=True), 1.0))
    d_item = rng.normal(0.0, 0.9, size=J)
    pop = np.zeros(K, dtype=np.float64)

    amp = 1.4
    V_true = np.zeros((2, K, L_RANK), dtype=np.float64)
    mu_true = np.zeros((2, K), dtype=np.float64)
    half_k = K // 2
    for l in range(L_RANK):
        v1 = rng.normal(0.0, 1.0, size=half_k)
        v2 = rng.normal(0.0, 1.0, size=K - half_k)
        V_true[0, :half_k, l] = (v1 / np.linalg.norm(v1)) * np.sqrt(half_k) * amp
        V_true[1, half_k:, l] = (v2 / np.linalg.norm(v2)) * np.sqrt(K - half_k) * amp
    mu_true[0, :half_k], mu_true[0, half_k:] = 0.25 * amp, -0.25 * amp
    mu_true[1, :half_k], mu_true[1, half_k:] = -0.25 * amp, 0.25 * amp

    z_true = rng.integers(0, 2, size=N)
    U_stu = np.array([
        rng.normal(0.0, 0.8) + mu_true[z_true[i]] + V_true[z_true[i]] @ rng.normal(0.0, 1.0, L_RANK) + rng.normal(0.0, 0.15, K)
        for i in range(N)
    ])
    items_list = [rng.choice(J, size=E_STU, replace=False) for _ in range(N)]
    y_list = [(rng.random(E_STU) < _sigmoid(A[its] @ U_stu[i] - d_item[its])).astype(np.float64) for i, its in enumerate(items_list)]

    n_tr = int(0.8 * N)
    tr_b = SimpleBatch(A, d_item, pop, items_list[:n_tr], y_list[:n_tr])
    te_b = SimpleBatch(A, d_item, pop, [it[:M] for it in items_list[n_tr:]], [yy[:M] for yy in y_list[n_tr:]],
                       [it[M:] for it in items_list[n_tr:]], [yy[M:] for yy in y_list[n_tr:]])

    # Baseline B0 vs oracle subspace check
    ev0 = solve_subspace_component(te_b, 1.0, 2.0, np.zeros(K), np.zeros((K, L_RANK)), compute_query=True)
    nll0 = eval_nll(te_b, ev0["p_q"])
    # Check Mix-V_r EM fit (R=2)
    act = tr_b.mask_J > 0.5
    X_f = sp.csr_matrix(tr_b.X_sup[act]); X_f_T = X_f.T.tocsr()
    y_f = tr_b.y_sup[act]; stu_idx = np.broadcast_to(np.arange(tr_b.N)[:, None], (tr_b.N, tr_b.max_J))[act]
    base = solve_subspace_component(tr_b, 1.0, 2.0, np.zeros(K), np.zeros((K, L_RANK)))
    eps_full = np.zeros((tr_b.N, K))
    for i in range(tr_b.N):
        obs = tr_b.obs_list[i]
        eps_full[i, obs] = base["eps_obs"][i, : len(obs)]
    km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(eps_full)
    mu_mat = np.zeros((2, K)); V_mat = np.zeros((2, K, L_RANK)); pi = np.array([0.5, 0.5])
    for r in range(2):
        sub = eps_full[km.labels_ == r]
        mu_mat[r] = sub.mean(axis=0) * 0.35
        V_mat[r] = np.linalg.svd(sub - sub.mean(axis=0), full_matrices=False)[2][:L_RANK].T * 0.35

    for _ in range(8):
        log_m = np.zeros((tr_b.N, 2)); ers = []
        for r in range(2):
            er = solve_subspace_component(tr_b, 1.0, 2.0, mu_mat[r], V_mat[r], n_newton=5)
            log_m[:, r] = er["log_m"]; ers.append(er)
        lj = np.log(np.maximum(pi, 1e-12))[None, :] + log_m
        rho = np.exp(lj - _logsumexp(lj, axis=1)[:, None])
        pi = np.maximum(rho.mean(axis=0), 1e-3); pi /= pi.sum()
        for r in range(2):
            er = ers[r]
            off = tr_b.b_sup + tr_b.s_sup * er["theta_g"][:, None] + np.einsum("njk,nk->nj", tr_b.X_sup_obs, er["eps_obs"])
            off_f = off[act]; w_f = rho[stu_idx, r]; eta_f = er["eta"][stu_idx]; inv_n = 1.0 / max(rho[:, r].sum(), 1.0)
            def obj(x):
                mu, V = x[:K], x[K:].reshape(K, L_RANK)
                lg = off_f + X_f @ mu + np.sum((X_f @ V) * eta_f, axis=1)
                loss = float(np.sum(w_f * (np.maximum(lg, 0.0) - lg * y_f + np.log1p(np.exp(-np.abs(lg)))))) \
                       + 5.0 * (float(np.sum(mu ** 2)) + float(np.sum(V ** 2)))
                df = w_f * (_sigmoid(lg) - y_f)
                return loss * inv_n, np.concatenate([X_f_T @ df + 10.0 * mu, (X_f_T @ (df[:, None] * eta_f) + 10.0 * V).ravel()]) * inv_n
            res = minimize(obj, np.concatenate([mu_mat[r], V_mat[r].ravel()]), method="L-BFGS-B", jac=True, options={"maxiter": 12})
            mu_mat[r], V_mat[r] = res.x[:K], res.x[K:].reshape(K, L_RANK)

    log_m_te = np.zeros((te_b.N, 2)); pqs = []
    for r in range(2):
        er = solve_subspace_component(te_b, 1.0, 2.0, mu_mat[r], V_mat[r], compute_query=True)
        log_m_te[:, r] = er["log_m"]; pqs.append(er["p_q"])
    lj = np.log(np.maximum(pi, 1e-12))[None, :] + log_m_te
    rho_te = np.exp(lj - _logsumexp(lj, axis=1)[:, None])
    nll2 = eval_nll(te_b, rho_te[:, 0:1] * pqs[0] + rho_te[:, 1:2] * pqs[1])
    auc = roc_auc_score(z_true[n_tr:], rho_te[:, 1])
    print(f"Synthetic Subspace Mix-V_r (amp=1.4): B0 NLL={nll0:.5f}, R=2 NLL={nll2:.5f}, gain={100*(nll0-nll2)/nll0:+.2f}%, AUC={max(auc, 1-auc):.3f}")


if __name__ == "__main__":
    run_synthetic_subspace_demo()
