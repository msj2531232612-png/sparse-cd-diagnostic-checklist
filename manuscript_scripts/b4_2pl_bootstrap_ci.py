# -*- coding: utf-8 -*-
"""
Full-pipeline student-cluster bootstrap CIs for 1-PL and 2-PL residual/anchor ratios
across matched min_n thresholds.
"""
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
from d6_crossfit_1000 import residual_a

SIG = mc.SIG


def safe_top1(R):
    obs = ~np.isnan(R)
    cnt = obs.sum(axis=0)
    mu = np.zeros(R.shape[1], dtype=np.float64)
    nz = cnt > 0
    mu[nz] = np.nanmean(R[:, nz], axis=0)
    Rc = np.where(obs, R - mu, 0.0)
    return float(np.linalg.svd(Rc, compute_uv=False)[0])


def _prep_user_index(d):
    n_u = d["n_u"]
    order = np.argsort(d["qu"], kind="mergesort")
    counts = np.bincount(d["qu"][order], minlength=n_u)
    starts = np.zeros(n_u + 1, dtype=np.int64)
    np.cumsum(counts, out=starts[1:])
    return {
        "n_u": n_u, "n_item": d["n_item"], "K": d["K"],
        "qit_s": d["qit"][order], "y_s": d["y"][order],
        "starts": starts, "kc_list": d["kc_list"],
    }


def _resample_students(u_idx_data, rng):
    n_u, starts = u_idx_data["n_u"], u_idx_data["starts"]
    sel_u = rng.integers(0, n_u, size=n_u)
    lens = starts[sel_u + 1] - starts[sel_u]
    tot = int(lens.sum())
    qu_b = np.repeat(np.arange(n_u, dtype=np.int64), lens)
    qit_b = np.empty(tot, dtype=np.int64)
    y_b = np.empty(tot, dtype=np.float64)
    pos = 0
    for u_orig, L in zip(sel_u, lens):
        s0, s1 = starts[u_orig], starts[u_orig + 1]
        qit_b[pos:pos + L] = u_idx_data["qit_s"][s0:s1]
        y_b[pos:pos + L] = u_idx_data["y_s"][s0:s1]
        pos += L
    uniq_items, qit_compact = np.unique(qit_b, return_inverse=True)
    kc_list, K = u_idx_data["kc_list"], u_idx_data["K"]
    if K == 406:
        kc_arr = np.array([kc_list[it][0] for it in uniq_items], dtype=np.int64)
        owner = np.arange(tot, dtype=np.int64)
        conc = kc_arr[qit_compact]
    else:
        item_kcs = [kc_list[it] for it in uniq_items]
        kc_lens = np.array([len(item_kcs[idx]) for idx in qit_compact], dtype=np.int64)
        owner = np.repeat(np.arange(tot, dtype=np.int64), kc_lens)
        conc = np.concatenate([np.asarray(item_kcs[idx], dtype=np.int64) - 1 for idx in qit_compact])
    return {
        "n_u": n_u, "n_item": len(uniq_items), "K": K, "qu": qu_b, "qit": qit_compact,
        "y": y_b, "ev_owner": owner, "ev_c": conc,
    }


def _worker(args):
    u_idx_data, n_rep, min_n, seed = args
    rng = np.random.default_rng(seed)
    res = []
    for _ in range(n_rep):
        d = _resample_students(u_idx_data, rng)
        th1, dv1 = mc.rasch_fit(d, iters=6)
        r1 = safe_top1(mc.residual_matrix(d, th1, dv1, min_n=min_n)[0])
        a = est_a(d, th1, dv1, iters=20, ridge=20.0)
        r2 = safe_top1(residual_a(d, th1, dv1, a, min_n=min_n))

        th_s1 = rng.normal(0.0, 1.0, d["n_u"])
        d1s = dict(d); d1s["y"] = (rng.random(len(d["y"])) < SIG(th_s1[d["qu"]] - dv1[d["qit"]])).astype(np.float64)
        ts1, dvs1 = mc.rasch_fit(d1s, iters=6)
        a1 = safe_top1(mc.residual_matrix(d1s, ts1, dvs1, min_n=min_n)[0])

        th_s2 = rng.normal(0.0, 1.0, d["n_u"])
        d2s = dict(d); d2s["y"] = (rng.random(len(d["y"])) < SIG(a[d["qit"]] * (th_s2[d["qu"]] - dv1[d["qit"]]))).astype(np.float64)
        ts2, dvs2 = mc.rasch_fit(d2s, iters=6)
        a2_sl = est_a(d2s, ts2, dvs2, iters=20, ridge=20.0)
        a2 = safe_top1(residual_a(d2s, ts2, dvs2, a2_sl, min_n=min_n))
        res.append((r1 / max(a1, 1e-9), r2 / max(a2, 1e-9)))
    return res


def run_dataset(d, min_n=1, n_boot=120, n_workers=4, seed=700):
    u_idx = _prep_user_index(d)
    per_w = n_boot // n_workers
    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        chunks = list(ex.map(_worker, [(u_idx, per_w, min_n, seed + w * 101) for w in range(n_workers)]))
    arr = np.asarray([r for ch in chunks for r in ch], dtype=np.float64)
    return {
        "dataset": d["name"], "min_n": min_n, "n_boot": int(arr.shape[0]),
        "ratio_1pl_ci95": [float(np.percentile(arr[:, 0], 2.5)), float(np.percentile(arr[:, 0], 97.5))],
        "ratio_2pl_ci95": [float(np.percentile(arr[:, 1], 2.5)), float(np.percentile(arr[:, 1], 97.5))],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=mc.default_data_root())
    parser.add_argument("--n-boot", type=int, default=120)
    parser.add_argument("--out", default="./results/b4_bootstrap.json")
    args = parser.parse_args()
    os.environ["EDM_DATA_DIR"] = args.data_dir
    out = {
        "kdd_min_n1": run_dataset(mc.load_kdd(), min_n=1, n_boot=args.n_boot, seed=700),
        "a0910_min_n1": run_dataset(mc.load_a0910(), min_n=1, n_boot=args.n_boot, seed=900),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
