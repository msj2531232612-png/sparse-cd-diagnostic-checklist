"""Command-line entry for the cold-start diagnostic checklist toolkit.

Examples
--------
    python run_audit.py --demo                         # 10-second synthetic run
    python run_audit.py --demo --out-dir ./results     # + write JSON report
    python run_audit.py --task positive --seeds 100 110
    python run_audit.py --task static --data-dir /path/to/data --dataset a0910
"""
import argparse
import json
import os
import time

import numpy as np

from audit_toolkit import static_gate as SG
from audit_toolkit import dynamic_gate as DG
from audit_toolkit import positive_control as PC
from audit_toolkit import mechanism as MX


# --------------------------------------------------------------------------
# synthetic demo (no data required)
# --------------------------------------------------------------------------
def run_demo(out_dir=None):
    t0 = time.time()
    report = {}

    # 1) static positive control: the detector must release under strong signal
    verdict_pos, summ_pos, _ = PC.static_positive_control(
        seeds=range(100, 104), var_true=(0.1, 9.0), N=400, J=80, n_per=60, Q=32)
    report["static_positive"] = dict(verdict=verdict_pos, **summ_pos)

    # 2) static null control: the detector must reject under no signal
    verdict_null, summ_null, _ = PC.static_positive_control(
        seeds=range(200, 204), var_true=(1.0, 1.0), N=400, J=80, n_per=60, Q=32)
    report["static_null"] = dict(verdict=verdict_null, **summ_null)

    # 3) D1 residual spectrum on a null synthetic set: the permutation null
    #    only breaks cross-concept correlation (it keeps the fit-noise floor),
    #    so the homogeneous anchor is the honest baseline.
    D = SG.generate_static(0, N=400, J=80, n_per=40, var_true=(1.0, 1.0))
    d, theta = MX.fit_rasch(D["items"], D["y"], J=80)
    concept = np.arange(80) % 10          # 10 synthetic concepts, 8 items each
    R = MX.residual_matrix(D["items"], D["y"], d, theta, concept, K=10)
    spec = MX.spectrum(R, n_perm=60, seed=0)
    anchor = MX.homogeneous_anchor(d, D["items"], concept, K=10, n_sim=10, seed=0)
    report["mechanism_D1"] = dict(
        real_top1=spec["topk"][0], null95_top1=spec["null95"][0],
        anchor_top1=anchor,
        ratio_null=round(spec["topk"][0] / spec["null95"][0], 3),
        ratio_anchor=round(spec["topk"][0] / anchor, 3))

    # 4) dynamic design gate on a small strongly-separated set
    design, ident, summ_dyn, _ = PC.dynamic_positive_control(
        seeds=range(300, 302), E_TR=60, E_FUT=20, G_TRUE=(0.02, 0.6),
        DT_MEAN=0.3, Q=24)
    report["dynamic"] = dict(**summ_dyn)

    report["elapsed_s"] = round(time.time() - t0, 1)
    report["verdicts"] = {
        "static_positive_RELEASE": report["static_positive"]["verdict"] == "RELEASE",
        "static_null_REJECT": report["static_null"]["verdict"] == "REJECT",
        "dynamic_design_gate": report["dynamic"]["design_gate"],
    }

    print("=" * 72)
    print("Cold-start diagnostic checklist  (synthetic demo)")
    print("=" * 72)
    print(f"[static positive control] verdict={report['static_positive']['verdict']}  "
          f"gain {report['static_positive']['gain_median']:.2f}%  "
          f"BIC-R2 {report['static_positive']['bic_picks_R2']}  "
          f"AUC {report['static_positive']['route_auc_median']:.3f}")
    print(f"[static null control]    verdict={report['static_null']['verdict']}  "
          f"gain {report['static_null']['gain_median']:.2f}%  "
          f"BIC-R2 {report['static_null']['bic_picks_R2']}  "
          f"AUC {report['static_null']['route_auc_median']:.3f}")
    print(f"[mechanism D1]           real top1 {report['mechanism_D1']['real_top1']:.2f} vs "
          f"perm-null95 {report['mechanism_D1']['null95_top1']:.2f} "
          f"(ratio {report['mechanism_D1']['ratio_null']}) vs "
          f"homog anchor {report['mechanism_D1']['anchor_top1']:.2f} "
          f"(ratio {report['mechanism_D1']['ratio_anchor']})")
    print(f"[dynamic design gate]    design_gate={report['dynamic']['design_gate']}  "
          f"ident={report['dynamic']['identifiability']}  "
          f"D5 {report['dynamic']['D5_gain']:.2f}% / MixTrue {report['dynamic']['MixTrue_gain']:.2f}%")
    print("note: the dynamic gate stays FAIL even under strong signal (oracle gain "
          "<0.5%) -- this is the manuscript's readout-insensitivity finding, not a bug.")
    print(f"elapsed {report['elapsed_s']}s")
    print("=" * 72)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "demo_report.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
        print(f"report -> {path}")
    return report


# --------------------------------------------------------------------------
# minimal real-data loader (long-format CSV: user_id, item_id, score)
# --------------------------------------------------------------------------
def load_long(path, concept_col=None):
    """Load long-format data into dense (items (N,n), y (N,n), mask (N,n)).

    Supports headered or headerless CSV with columns user_id, item_id, score
    (and optionally concept_id). Returns None if the file cannot be parsed.
    """
    import csv
    from collections import defaultdict, Counter
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    if not raw_rows:
        return None

    first = raw_rows[0]
    has_header = False
    try:
        int(first[0]); int(first[1]); float(first[2])
    except (ValueError, IndexError):
        has_header = True

    c_idx = None
    if has_header:
        header = [h.strip() for h in first]
        data_rows = raw_rows[1:]
        if isinstance(concept_col, str):
            if concept_col in header:
                c_idx = header.index(concept_col)
            else:
                raise ValueError(f"concept_col '{concept_col}' not found in header {header}")
        elif isinstance(concept_col, int):
            c_idx = concept_col
        elif "concept_id" in header:
            c_idx = header.index("concept_id")
        elif len(header) >= 4:
            c_idx = 3
    else:
        data_rows = raw_rows
        if isinstance(concept_col, int):
            c_idx = concept_col
        elif len(first) >= 4 and concept_col is None:
            c_idx = 3

    for r in data_rows:
        if len(r) < 3:
            continue
        u, it, s = int(r[0]), int(r[1]), float(r[2])
        c = int(r[c_idx]) if (c_idx is not None and len(r) > c_idx and r[c_idx].strip() != "") else 0
        rows.append((u, it, s, c))
    if not rows:
        return None
    by_user = defaultdict(list)
    for u, it, s, c in rows:
        by_user[u].append((it, s, c))
    users = sorted(by_user)
    n = max(len(v) for v in by_user.values())
    items = np.zeros((len(users), n), int)
    y = np.zeros((len(users), n), float)
    mask = np.zeros((len(users), n), bool)
    concept = np.zeros((len(users), n), int)
    for i, u in enumerate(users):
        for k, (it, s, c) in enumerate(by_user[u]):
            items[i, k], y[i, k], concept[i, k] = it, s, c
            mask[i, k] = True
    # 1-D item -> concept map (majority vote over observed responses)
    item_concepts = defaultdict(Counter)
    for _u, it, _s, c in rows:
        item_concepts[it][c] += 1
    n_items = max(item_concepts) + 1
    concept_of_item = np.zeros(n_items, int)
    for it, cnt in item_concepts.items():
        concept_of_item[it] = cnt.most_common(1)[0][0]
    return dict(items=items, y=y, mask=mask, concept=concept,
                concept_of_item=concept_of_item, n_users=len(users))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Cold-start feasibility-gate audit")
    ap.add_argument("--demo", action="store_true", help="run the synthetic demo")
    ap.add_argument("--task", choices=["demo", "positive", "static", "dynamic", "mechanism"],
                    default="demo")
    ap.add_argument("--data-dir", default=None, help="directory holding dataset files")
    ap.add_argument("--dataset", default=None, help="dataset name / CSV filename")
    ap.add_argument("--concept-col", default=None, help="concept column name or 0-based index")
    ap.add_argument("--n-fit", type=int, default=None, help="support responses per eval user (static)")
    ap.add_argument("--seed", type=int, default=42, help="train/eval user split seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--Q", type=int, default=48, help="Gauss-Hermite points")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    if args.demo or args.task == "demo":
        run_demo(out_dir=args.out_dir)
        return

    if args.task == "positive":
        seeds = args.seeds or range(100, 112)
        verdict, summ, _ = PC.static_positive_control(seeds=seeds, Q=args.Q)
        print(f"static positive control: verdict={verdict}")
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        return

    if args.task == "dynamic":
        if args.data_dir or args.dataset:
            print(json.dumps({
                "verdict": "insufficient evidence",
                "reason": "real-log dynamic audit requires timestamp/delta_t support and is not implemented in this CLI prototype; omit --data-dir/--dataset for the synthetic dynamic control."
            }, ensure_ascii=False, indent=1))
            return
        seeds = args.seeds or range(200, 212)
        design, ident, summ, _ = PC.dynamic_positive_control(seeds=seeds, Q=args.Q)
        print(f"dynamic positive control: design_gate={design} identifiability={ident}")
        print(json.dumps(summ, ensure_ascii=False, indent=1))
        return

    # real-data tasks require --data-dir + --dataset
    if not args.data_dir or not args.dataset:
        raise SystemExit("--task static/mechanism requires --data-dir and --dataset")
    path = os.path.join(args.data_dir, args.dataset)
    ccol = int(args.concept_col) if (args.concept_col is not None and args.concept_col.isdigit()) else args.concept_col
    D = load_long(path, concept_col=ccol)
    if D is None:
        raise SystemExit(f"could not load {path} (need columns user_id, item_id, score)")
    n_users = D["n_users"]
    if n_users < 2:
        raise SystemExit(f"need at least 2 users for train/eval isolation (got {n_users})")

    # Record the exact input artefact (basename + sha256) and its size so a
    # report can be audited without shipping the data itself.
    import hashlib as _hl
    _h = _hl.sha256()
    with open(path, "rb") as _f:
        for _c in iter(lambda: _f.read(1 << 20), b""):
            _h.update(_c)
    input_meta = {
        "input_basename": os.path.basename(path),
        "input_sha256": _h.hexdigest(),
        "n_rows": int(len(D["items"])),
        "n_users": int(n_users),
    }

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_users)
    n_train = max(1, min(n_users - 1, int(round(0.65 * n_users))))
    tr_idx, ev_idx = perm[:n_train], perm[n_train:]

    J = int(D["items"].max()) + 1
    d, _theta_tr = MX.fit_rasch(D["items"][tr_idx], D["y"][tr_idx], J, mask=D["mask"][tr_idx])

    if args.task == "static":
        ev_items, ev_y, ev_mask = D["items"][ev_idx], D["y"][ev_idx], D["mask"][ev_idx]
        if args.n_fit is not None:
            n_fit = args.n_fit
        else:
            min_ev_len = int(ev_mask.sum(1).min())
            n_fit = max(1, min(20, (min_ev_len * 2) // 3))
        res = SG.audit_static(d, ev_items, ev_y, n_fit=n_fit, Q=args.Q, mask=ev_mask)
        res["verdict"] = SG.classify_static_verdict(res)
        res["n_train_users"] = int(len(tr_idx))
        res["meta"] = dict(input_meta, hyper=dict(
            task="static", n_fit=n_fit, Q=args.Q, seed=args.seed,
            train_user_fraction=0.65, n_train_users=int(len(tr_idx)),
            n_eval_users=int(len(ev_idx)), bootstrap_resamples=500,
            bootstrap_seed=0, concept_col=ccol))
        print(json.dumps(res, ensure_ascii=False, indent=1))
    else:  # mechanism
        ev_items, ev_y, ev_mask = D["items"][ev_idx], D["y"][ev_idx], D["mask"][ev_idx]
        theta_ev = MX.fit_theta_frozen(ev_items, ev_y, d, mask=ev_mask)
        K = int(D["concept_of_item"].max()) + 1
        R = MX.residual_matrix(ev_items, ev_y, d, theta_ev, D["concept_of_item"], K, mask=ev_mask)
        spec = MX.spectrum(R, n_perm=200, seed=0)
        anchor = MX.homogeneous_anchor(d, ev_items, D["concept_of_item"], K,
                                       n_sim=20, seed=0, mask=ev_mask, frozen_d=True)
        spec["anchor_top1"] = anchor
        spec["ratio_anchor"] = round(spec["topk"][0] / anchor, 3)
        spec["n_train_users"] = int(len(tr_idx))
        spec["n_eval_users"] = int(len(ev_idx))
        spec["meta"] = dict(input_meta, hyper=dict(
            task="mechanism", Q=args.Q, seed=args.seed, n_perm=200,
            train_user_fraction=0.65, n_train_users=int(len(tr_idx)),
            n_eval_users=int(len(ev_idx)), concept_col=ccol))
        print(json.dumps(spec, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
