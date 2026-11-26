# -*- coding: utf-8 -*-
"""
Standalone student-level 65/15/20 three-way split loaders for ASSISTments a0910
and KDD Cup 2010 algebra_probe.npz (zero external path dependencies).
Set EDM_DATA_DIR or pass data_dir explicitly.
"""
import ast
import os
import numpy as np
import pandas as pd


def default_data_dir():
    return os.environ.get("EDM_DATA_DIR", "./data")


def load_a0910_3way(seed=42, data_dir=None):
    base = data_dir or default_data_dir()
    d = os.path.join(base, "a0910")
    item_df = pd.read_csv(os.path.join(d, "item.csv"))
    all_df = pd.concat(
        [pd.read_csv(os.path.join(d, fn)) for fn in ["train.csv", "valid.csv", "test.csv"]],
        ignore_index=True,
    )
    all_df["score"] = (all_df["score"] > 0.5).astype(np.float32)

    item_ids = sorted(item_df["item_id"].unique())
    item2idx = {it: i for i, it in enumerate(item_ids)}
    all_kcs, item_kc_map = set(), {}
    for _, r in item_df.iterrows():
        kcs = ast.literal_eval(str(r["knowledge_code"]))
        item_kc_map[item2idx[r["item_id"]]] = kcs
        all_kcs.update(kcs)
    kc_list = sorted(all_kcs)
    kc2idx = {kc: i for i, kc in enumerate(kc_list)}
    n_items, K = len(item_ids), len(kc_list)
    Q = np.zeros((n_items, K), dtype=np.float32)
    for it_i, kcs in item_kc_map.items():
        for kc in kcs:
            Q[it_i, kc2idx[kc]] = 1.0

    all_df = all_df[all_df["item_id"].isin(item2idx)].copy()
    all_df["item_idx"] = all_df["item_id"].map(item2idx)

    users = sorted(all_df["user_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_tr, n_tu = int(0.65 * len(users)), int(0.15 * len(users))
    tr_u_set = set(users[:n_tr])
    tu_u_set = set(users[n_tr : n_tr + n_tu])
    te_u_set = set(users[n_tr + n_tu :])

    tr_df = all_df[all_df["user_id"].isin(tr_u_set)].copy()
    seen_items = set(tr_df["item_idx"].unique())

    def prep_eval_df(u_set):
        sub = all_df[all_df["user_id"].isin(u_set) & all_df["item_idx"].isin(seen_items)].copy()
        vc = sub.groupby("user_id").size()
        keep = sorted(vc[vc >= 30].index.tolist())
        sub = sub[sub["user_id"].isin(keep)].copy()
        sub["u_idx"] = sub["user_id"].map({u: i for i, u in enumerate(keep)})
        return sub, len(keep)

    tu_df, n_tu_u = prep_eval_df(tu_u_set)
    te_df, n_te_u = prep_eval_df(te_u_set)
    tr_ulist = sorted(tr_df["user_id"].unique())
    tr_df["u_idx"] = tr_df["user_id"].map({u: i for i, u in enumerate(tr_ulist)})
    return (
        (tr_df["u_idx"].values, tr_df["item_idx"].values, tr_df["score"].values),
        tu_df, te_df, Q, len(tr_ulist), n_tu_u, n_te_u, n_items, K,
    )


def load_kdd_algebra_3way(seed=42, data_dir=None):
    base = data_dir or default_data_dir()
    path = os.path.join(base, "kddcup2010", "algebra_probe.npz")
    z = np.load(path, allow_pickle=False)
    u = z["u"].astype(np.int64)
    it = z["it"].astype(np.int64)
    y = (z["y"] > 0.5).astype(np.float32)
    Q = z["Q"].astype(np.float32)
    n_items, K = Q.shape

    df = pd.DataFrame({"user_id": u, "item_idx": it, "score": y})
    df = df.drop_duplicates(subset=["user_id", "item_idx"], keep="first").reset_index(drop=True)

    users = np.array(sorted(df["user_id"].unique()), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n_u = len(users)
    n_tr, n_tu = int(0.65 * n_u), int(0.15 * n_u)
    tr_u_set = set(users[:n_tr].tolist())
    tu_u_set = set(users[n_tr : n_tr + n_tu].tolist())
    te_u_set = set(users[n_tr + n_tu :].tolist())

    tr_df = df[df["user_id"].isin(tr_u_set)].copy()
    seen_items = set(tr_df["item_idx"].unique().tolist())

    def prep_eval_df(u_set):
        sub = df[df["user_id"].isin(u_set) & df["item_idx"].isin(seen_items)].copy()
        vc = sub.groupby("user_id").size()
        keep = sorted(vc[vc >= 30].index.tolist())
        sub = sub[sub["user_id"].isin(keep)].copy()
        sub["u_idx"] = sub["user_id"].map({uid: i for i, uid in enumerate(keep)})
        return sub, len(keep)

    tu_df, n_tu_u = prep_eval_df(tu_u_set)
    te_df, n_te_u = prep_eval_df(te_u_set)
    tr_ulist = sorted(tr_df["user_id"].unique().tolist())
    tr_df["u_idx"] = tr_df["user_id"].map({uid: i for i, uid in enumerate(tr_ulist)})
    return (
        (tr_df["u_idx"].values, tr_df["item_idx"].values, tr_df["score"].values),
        tu_df, te_df, Q, len(tr_ulist), n_tu_u, n_te_u, n_items, K,
    )
