#!/usr/bin/env python3
"""Unified CLI for RF-Diffusion evaluation pipeline.

Usage:
    python -m src.cli wifi --num-samples 5 --gpu 0
    python -m src.cli mimo --num-samples 3 --gpu 0
    python -m src.cli efficiency --num-samples 2 --gpu 0
    python -m src.cli plot
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _run(script_name: str, extra_args: list[str] | None = None) -> int:
    """Delegate to a scripts/<script_name>.py entry point."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def cmd_wifi(args: argparse.Namespace) -> int:
    """Wi-Fi CSI generation and evaluation."""
    extra = []
    if args.num_samples is not None:
        extra += ["--num-samples", str(args.num_samples)]
    extra += ["--gpu", str(args.gpu)]
    if args.strategy:
        extra += ["--strategy", args.strategy]
    return _run("run_wifi_sampling.py", extra)


def cmd_mimo(args: argparse.Namespace) -> int:
    """5G MIMO channel estimation."""
    extra = []
    if args.num_samples is not None:
        extra += ["--num-samples", str(args.num_samples)]
    extra += ["--gpu", str(args.gpu)]
    return _run("run_5g_mimo.py", extra)


def cmd_efficiency(args: argparse.Namespace) -> int:
    """Quality vs inference time trade-off analysis."""
    extra = ["--num-samples", str(args.num_samples), "--gpu", str(args.gpu)]
    if args.mode:
        extra += ["--mode", args.mode]
    return _run("run_efficiency.py", extra)


def cmd_plot(args: argparse.Namespace) -> int:
    """Generate publication-quality figures from evaluation results."""
    return _run("plot_results.py")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run wifi + mimo evaluation end-to-end."""
    ret = cmd_wifi(args)
    if ret != 0:
        return ret
    return cmd_mimo(args)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rf-diffusion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_wifi = sub.add_parser("wifi", help="Wi-Fi CSI generation and evaluation")
    p_wifi.add_argument("--num-samples", type=int, default=None)
    p_wifi.add_argument("--gpu", type=int, default=0)
    p_wifi.add_argument("--strategy", choices=["native", "fast", "full_reverse", "all"],
                        default="native")
    p_wifi.set_defaults(func=cmd_wifi)

    p_mimo = sub.add_parser("mimo", help="5G MIMO channel estimation")
    p_mimo.add_argument("--num-samples", type=int, default=None)
    p_mimo.add_argument("--gpu", type=int, default=0)
    p_mimo.set_defaults(func=cmd_mimo)

    p_eff = sub.add_parser("efficiency", help="Quality vs inference time analysis")
    p_eff.add_argument("--num-samples", type=int, default=3)
    p_eff.add_argument("--gpu", type=int, default=0)
    p_eff.add_argument("--mode", choices=["all", "native", "full_reverse"], default="all")
    p_eff.set_defaults(func=cmd_efficiency)

    sub.add_parser("plot", help="Generate publication-quality figures").set_defaults(func=cmd_plot)

    p_eval = sub.add_parser("evaluate", help="Run wifi + mimo end-to-end")
    p_eval.add_argument("--num-samples", type=int, default=None)
    p_eval.add_argument("--gpu", type=int, default=0)
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
