# -*- coding: utf-8 -*-
"""
Shared sparse mechanism-analysis utilities for residual matrix construction,
1-PL Rasch MAP/Newton fitting, column-permutation nulls, and coverage metrics.
Configurable data root via EDM_DATA_DIR environment variable or function argument.
"""
import csv
import io
import json
import os
import numpy as np

SIG = lambda z: 0.5 * (1.0 + np.tanh(0.5 * z))


def default_data_root():
    return os.environ.get("EDM_DATA_DIR", "./data")


def load_a0910(root=None):
    if root is None:
        root = os.path.join(default_data_root(), "a0910")
    item2kc = {}
    with io.open(os.path.join(root, "item.csv"), encoding="utf-8", errors="replace") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            item2kc[int(row[0])] = [int(x) for x in json.loads(row[1])]
    ev = []
    for fn in ["train.csv", "valid.csv", "test.csv"]:
        with io.open(os.path.join(root, fn), encoding="utf-8", errors="replace") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                try:
                    q = int(row[1])
                    yv = float(row[2])
                    u = int(row[0])
                except Exception:
                    continue
                if yv in (0.0, 1.0) and q in item2kc:
                    ev.append((u, q, yv))
    uids = sorted({e[0] for e in ev})
    umap = {v: k for k, v in enumerate(uids)}
    items = sorted({e[1] for e in ev})
    imap = {v: k for k, v in enumerate(items)}
    qu = np.array([umap[e[0]] for e in ev], dtype=np.int64)
    qit = np.array([imap[e[1]] for e in ev], dtype=np.int64)
    y = np.array([e[2] for e in ev], dtype=np.float64)
    kc_list = [sorted(set(item2kc[it])) for it in items]
    owner, conc = [], []
    for e in range(len(ev)):
        for c in kc_list[qit[e]]:
            owner.append(e)
            conc.append(c - 1)
    return dict(
        name="a0910",
        qu=qu,
        qit=qit,
        y=y,
        ev_owner=np.asarray(owner, dtype=np.int64),
        ev_c=np.asarray(conc, dtype=np.int64),
        K=123,
        n_u=len(uids),
        n_item=len(items),
        kc_list=kc_list,
    )


def load_kdd(root=None):
    if root is None:
        root = os.path.join(default_data_root(), "kddcup2010", "algebra_probe.npz")
    z = np.load(root)
    u = z["u"].astype(np.int64)
    it = z["it"].astype(np.int64)
    y = z["y"].astype(np.float64)
    Q = z["Q"]
    uids = np.unique(u)
    items = np.unique(it)
    umap = {v: k for k, v in enumerate(uids)}
    imap = {v: k for k, v in enumerate(items)}
    qu = np.array([umap[v] for v in u], dtype=np.int64)
    qit = np.array([imap[v] for v in it], dtype=np.int64)
    kc_of_item = np.argmax(Q, axis=1)
    owner = np.arange(len(y), dtype=np.int64)
    conc = kc_of_item[qit].astype(np.int64)
    kc_list = [[int(c)] for c in kc_of_item]
    return dict(
        name="kdd_algebra",
        qu=qu,
        qit=qit,
        y=y,
        ev_owner=owner,
        ev_c=conc,
        K=int(Q.shape[1]),
        n_u=len(uids),
        n_item=len(items),
        kc_list=kc_list,
    )


def rasch_fit(d, iters=6):
    qu, qit, y = d["qu"], d["qit"], d["y"]
    n_u, n_item = d["n_u"], d["n_item"]
    theta = np.zeros(n_u)
    dv = np.zeros(n_item)
    for _ in range(iters):
        p = SIG(theta[qu] - dv[qit])
        w = p * (1 - p)
        num = np.bincount(qit, weights=y - p, minlength=n_item)
        den = np.bincount(qit, weights=w, minlength=n_item) + 1e-9
        dv -= num / den
        for _ in range(2):
            p = SIG(theta[qu] - dv[qit])
            w = p * (1 - p)
            num = np.bincount(qu, weights=y - p, minlength=n_u)
            den = np.bincount(qu, weights=w, minlength=n_u) + 1.0
            theta += num / den
    return theta, dv


def residual_matrix(d, theta, dv, min_n=1):
    K = d["K"]
    n_u = d["n_u"]
    res = d["y"] - SIG(theta[d["qu"]] - dv[d["qit"]])
    flat = d["qu"][d["ev_owner"]] * K + d["ev_c"]
    num = np.bincount(flat, weights=res[d["ev_owner"]], minlength=n_u * K)
    cnt = np.bincount(flat, minlength=n_u * K).astype(np.float64)
    R = np.where(cnt >= min_n, num / np.maximum(cnt, 1), np.nan).reshape(n_u, K)
    return R, cnt.reshape(n_u, K)
