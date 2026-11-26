# -*- coding: utf-8 -*-
"""1,000-shuffle column-permutation null across 3 seeds (7, 42, 99) and CV(a_j) slope check."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import concurrent.futures as cf
import json
import argparse
import numpy as np

import mechanism_common as mc

SIG = mc.SIG


def est_a(d, theta, dv, iters=20, ridge=20.0):
    qu, qit, y = d["qu"], d["qit"], d["y"]
    zd = theta[qu] - dv[qit]
    a = np.ones(d["n_item"])
    for _ in range(iters):
        p = SIG(a[qit] * zd)
        g = np.bincount(qit, weights=(y - p) * zd, minlength=d["n_item"])
        h = np.bincount(qit, weights=p * (1 - p) * zd ** 2, minlength=d["n_item"]) + ridge
        a = np.clip(a + g / h, 0.05, 5.0)
    return a


def _run_seed_chunk(args):
    Rc, obs_idx, seed, n_perm, v_init = args
    rng = np.random.default_rng(seed)
    tops = np.empty(n_perm, dtype=np.float64)
    n_iter = 140
    for p in range(n_perm):
        Rp = Rc.copy()
        for c, idx_c in enumerate(obs_idx):
            if len(idx_c) > 1:
                Rp[idx_c, c] = Rp[rng.permutation(idx_c), c]
        v = v_init.copy()
        prev = 0.0
        for it in range(n_iter):
            u = Rp @ v
            nu = float(np.linalg.norm(u))
            if nu < 1e-12:
                nu = 0.0
                break
            v = Rp.T @ u
            nv = float(np.linalg.norm(v))
            if nv < 1e-12:
                nu = 0.0
                break
            v /= nv
            if abs(nu - prev) < 1e-7 and it >= 25:
                break
            prev = nu
        tops[p] = nu
    return seed, tops


def check_perm(d, n_perm=1000, seeds=(7, 42, 99)):
    th, dv = mc.rasch_fit(d)
    R, _ = mc.residual_matrix(d, th, dv, min_n=1)
    obs = ~np.isnan(R)
    Rc = np.where(obs, R - np.nanmean(R, axis=0), 0.0)
    real_s1 = float(np.linalg.svd(Rc, compute_uv=False)[0])
    v_init = np.random.default_rng(123).normal(size=Rc.shape[1])
    v_init /= np.linalg.norm(v_init)
    obs_idx = [np.where(obs[:, c])[0] for c in range(Rc.shape[1])]

    per_seed = []
    with cf.ProcessPoolExecutor(max_workers=3) as ex:
        for sd, tops in ex.map(_run_seed_chunk, [(Rc, obs_idx, sd, n_perm, v_init) for sd in seeds]):
            p95 = float(np.percentile(tops, 95))
            pval = float((1 + np.sum(tops >= real_s1)) / (1 + n_perm))
            per_seed.append({
                "seed": int(sd),
                "p95": p95,
                "p50": float(np.median(tops)),
                "ratio_real_to_p95": float(real_s1 / p95),
                "p_value": pval,
            })

    a = est_a(d, th, dv)
    cv_a = float(np.std(a) / np.mean(a))
    return {
        "dataset": d["name"],
        "real_1pl_s1": real_s1,
        "n_perm": n_perm,
        "per_seed": per_seed,
        "a_mean": float(np.mean(a)),
        "a_std": float(np.std(a)),
        "cv_a": cv_a,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=mc.default_data_root())
    parser.add_argument("--out", default="./results/perm_1000.json")
    args = parser.parse_args()

    os.environ["EDM_DATA_DIR"] = args.data_dir
    out = {
        "kdd_algebra": check_perm(mc.load_kdd()),
        "a0910": check_perm(mc.load_a0910()),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
