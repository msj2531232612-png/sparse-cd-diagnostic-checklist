"""audit_toolkit: a pre-deployment feasibility gate for cold-start cognitive diagnosis.

This package reproduces the matched-control audit in the manuscript
"A Pre-Deployment Feasibility Gate for Cold-Start Cognitive Diagnosis".
It is self-contained: no absolute paths, no external data required (a synthetic
`--demo` runs end-to-end), and it reads real data only through explicit
`--data-dir` / `--dataset` arguments.

Modules:
- core_laplace      Gauss--Hermite quadrature, Rasch log-marginals, metrics
- static_gate       static mixture screen (single-ridge vs. mixture baselines)
- dynamic_gate      longitudinal forgetting screen (temporal horizon check)
- positive_control  detector-validity checks (static + dynamic)
- mechanism         D1 residual-spectrum diagnostic (SVD + permutation null)
"""
__version__ = "1.0.0"
