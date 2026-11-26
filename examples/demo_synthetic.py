"""Minimal synthetic run — no dataset required.

    python examples/demo_synthetic.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from run_audit import run_demo

if __name__ == "__main__":
    run_demo(out_dir=os.path.join(os.path.dirname(__file__), "..", "results"))
