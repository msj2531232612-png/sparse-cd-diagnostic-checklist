# Cold-Start Diagnostic Screen for Sparse Cognitive Diagnosis

Reference toolkit and manuscript reproduction scripts for:

> *How Much Signal Do Cold-Start Mixture Models Need? Item-Slope Confounding and Identifiability Limits in Sparse Cognitive Diagnosis*

This archive provides both a self-contained CLI diagnostic package (`run_audit.py` + `audit_toolkit/`) using a portable scalar Rasch readout and the anonymized full-manuscript pipelines (`manuscript_scripts/`) for reproducing the hierarchical readout (`Eq. (1)`), dichotomous Mixture Rasch (`Rost, 1990`), mode-specific low-rank subspace mixture (`V_r, mu_r`), spectral permutation nulls, and cross-fitting bootstrap analyses on public benchmarks (`a0910`, `KDD Cup 2010 algebra`, `XES3G5M`).

No raw student data or personal identifiers are bundled. Data paths default to environment variable `EDM_DATA_DIR` (or `--data-dir`).

---

## 1. Quick Start (Zero External Data)

```bash
pip install -r requirements.txt
python run_audit.py --demo
```

Required dependencies: `numpy`, `scipy`, `scikit-learn` (and `pandas` / `torch` only when running full hierarchical readout reproduction scripts on external datasets).

`--demo` executes four fast synthetic verification checks (~2 s):
1. **Static Positive Control**: synthetic 2-mode variance structure $\rightarrow$ `RELEASE`.
2. **Static Null Control**: homogeneous cohort $\rightarrow$ `REJECT`.
3. **Residual Spectrum Diagnostic**: singular value comparison against column-permutation null.
4. **Dynamic Design Sensitivity**: comparison across oracle, mixture, and static temporal decay arms.

---

## 2. Running the Diagnostic Toolkit on CSV Logs

Provide a long-format CSV with columns `user_id, item_id, score` (and optional `concept_id`):

```bash
python run_audit.py --task static --data-dir ./your_data --dataset a0910.csv
python run_audit.py --task mechanism --data-dir ./your_data --dataset a0910.csv --concept-col concept_id
```

Outputs include JSON summaries (`--out-dir ./results`), support-conditioned relative NLL differences, paired student bootstrap intervals, permutation null $p$-values, and three-state verdicts (`proceed to further evaluation`, `insufficient evidence`, `no improvement in tested configuration`).

---

## 3. Manuscript Scripts (`manuscript_scripts/`)

After setting `EDM_DATA_DIR` to a folder containing `a0910/`, `kddcup2010/algebra_probe.npz`, and `xes3g5m/` (from public educational data repositories):

- `manuscript_scripts/d1_1000_perm_check.py`: 1,000-shuffle column-permutation null across seeds 7, 42, 99.
- `manuscript_scripts/d6_crossfit_1000.py`: 1,000-draw 50/50 student cross-fitting + empirical $\hat{\sigma}_\theta$ dispersion check.
- `manuscript_scripts/b4_2pl_bootstrap_ci.py`: End-to-end student-cluster bootstrap CIs for 1-PL distortion ($2.15\times$) and 2-PL residual-anchor calibration.
- `manuscript_scripts/mixture_rasch_a1.py`: Dichotomous Mixture Rasch evaluation on 65/15/20 real splits and synthetic difficulty-shift MDE sweep ($\sigma_\Delta \in [0, 1.75]$).
- `manuscript_scripts/mode_subspace_vr_a3.py`: Mode-specific rank-$L$ concept subspace mixture ($V_r, \mu_r$) evaluation on real splits and planted orthogonal subspace MDE sweep.

All scripts are released under the MIT License (`LICENSE`).
