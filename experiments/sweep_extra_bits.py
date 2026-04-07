"""
sweep_extra_bits.py
-------------------
Sweeps extra_bits (0..17, configurable step) for the ipt_numba_exp functional
model and measures final action output RMSE vs the unpatched baseline for each
value.

The baseline (unpatched bfloat16) is run **once** and reused across all sweep
points, since it is independent of extra_bits.

Usage:
    OPENPI_DIR=/scratch/chloe.wong/openpi \\
    CUDA_VISIBLE_DEVICES=0 \\
    /scratch/chloe.wong/envs/pi0/bin/python experiments/sweep_extra_bits.py \\
        --label sweep_eb \\
        --extra-bits-min 0 --extra-bits-max 17 --extra-bits-step 4 \\
        --n-obs 2 --steps 5 \\
        --checkpoint-dir /scratch/chloe.wong/data/pi05_base \\
        --config pi05_droid_jointpos_polaris
"""

from __future__ import annotations

import argparse
import csv
import datetime
import math
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from pi0_inout._jax_stubs import inject as _inject_jax_stubs
_inject_jax_stubs()

from pi0_inout import (
    QuantFormat, QuantGroup,
    StatsTracker,
    patch_model, unpatch_model,
    patch_attn_sdpa, unpatch_attn_sdpa,
    patch_vector_ops, unpatch_vector_ops,
    register_functional_model,
    set_fp8_mode,
)
from pi0_inout.serve_quant import _load_norm_stats, Pi0PyTorchPolicy


from funct_models_ipt.ipt_numba_exp.ipt_rtl_linear import IPTLinearRTLFunction as IPTNumbaExpLinear

PASSTHROUGH = "passthrough"

# ---------------------------------------------------------------------------
# Helpers (mirrored from run_output_eval.py)
# ---------------------------------------------------------------------------

def _fmt_or_passthrough(s: str) -> Optional[QuantFormat]:
    if s == PASSTHROUGH:
        return None
    return QuantFormat(s)


def _make_dummy_obs(rng: np.random.Generator) -> dict:
    return {
        "observation/exterior_image_1_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left":      rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/joint_position":        rng.random(7).astype(np.float32),
        "observation/gripper_position":      rng.random(1).astype(np.float32),
        "prompt": "Grab the object",
    }


def _start_heartbeat(
    label: str,
    t0: float,
    stop_event: threading.Event,
    interval_s: int = 30,
    mx_tracker: Optional[StatsTracker] = None,
) -> threading.Thread:
    def _loop():
        while not stop_event.wait(timeout=interval_s):
            elapsed = time.monotonic() - t0
            calls = mx_tracker._seq if mx_tracker else 0
            print(
                f"  [heartbeat] {label}  elapsed={datetime.timedelta(seconds=int(elapsed))}"
                f"  layer_calls={calls}",
                flush=True,
            )
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def _collect_actions(
    policy: Pi0PyTorchPolicy,
    observations: list,
    num_steps: int,
    label: str,
    t0: float,
) -> list[torch.Tensor]:
    actions = []
    n = len(observations)
    stop = threading.Event()
    _start_heartbeat(label, t0, stop)
    for i, obs in enumerate(observations):
        print(f"\n[{label}] obs {i+1}/{n} ({num_steps} diffusion steps)...", flush=True)
        obs_t0 = time.monotonic()
        result = policy.infer(obs)
        act = torch.from_numpy(result["actions"])
        print(f"[{label}] obs {i+1}/{n} done in {time.monotonic()-obs_t0:.1f}s", flush=True)
        actions.append(act)
    stop.set()
    return actions


# ---------------------------------------------------------------------------
# Quantized pass only (no baseline re-run)
# ---------------------------------------------------------------------------

def _run_quantized_pass(
    policy: Pi0PyTorchPolicy,
    observations: list,
    active_groups: set[QuantGroup],
    mx_input_fmt: Optional[QuantFormat],
    mx_output_fmt: Optional[QuantFormat],
    vec_input_fmt: Optional[QuantFormat],
    vec_output_fmt: Optional[QuantFormat],
    functional_model_name: str,
    num_steps: int,
    t0: float,
) -> list[torch.Tensor]:
    model = policy.model

    _mx_in  = mx_input_fmt  or QuantFormat.BFLOAT16
    _mx_out = mx_output_fmt or QuantFormat.BFLOAT16
    _vi     = vec_input_fmt  or QuantFormat.BFLOAT16
    _vo     = vec_output_fmt or QuantFormat.BFLOAT16

    from pi0_inout.functional_models import get_functional_model_factory
    fm_factory = get_functional_model_factory(functional_model_name)

    mx_tracker = StatsTracker()
    patch_model(
        model,
        mx_input_fmt=_mx_in,
        mx_output_fmt=_mx_out,
        tracker=mx_tracker,
        active_groups=active_groups,
        functional_model_factory=fm_factory,
    )
    attn_handles = patch_attn_sdpa(
        model,
        active_groups=active_groups,
        mx_input_fmt=_mx_in,
        mx_output_fmt=_mx_out,
        tracker=mx_tracker,
    )
    vec_handles, vec_ctx = patch_vector_ops(
        model,
        active_groups=active_groups,
        vec_input_fmt=_vi,
        vec_output_fmt=_vo,
        tracker=StatsTracker(),
    )

    quant_actions: list[torch.Tensor] = []
    n = len(observations)
    stop = threading.Event()
    _start_heartbeat(functional_model_name, t0, stop, mx_tracker=mx_tracker)
    with vec_ctx:
        for i, obs in enumerate(observations):
            print(f"\n[{functional_model_name}] obs {i+1}/{n} ({num_steps} diffusion steps)...", flush=True)
            obs_t0 = time.monotonic()
            result = policy.infer(obs)
            act = torch.from_numpy(result["actions"])
            print(f"[{functional_model_name}] obs {i+1}/{n} done in {time.monotonic()-obs_t0:.1f}s", flush=True)
            quant_actions.append(act)
    stop.set()

    unpatch_model(model)
    unpatch_attn_sdpa(attn_handles)
    unpatch_vector_ops(vec_handles)

    return quant_actions


# ---------------------------------------------------------------------------
# RMSE helpers (mirrored from run_output_eval.py)
# ---------------------------------------------------------------------------

_ACTION_RMSE_FIELDS = ["obs_idx", "rmse", "ref_rms", "rel_rmse"]


def _write_action_rmse(
    path: Path,
    baseline_actions: list[torch.Tensor],
    quant_actions: list[torch.Tensor],
) -> tuple[float, float]:
    rows = []
    total_se = total_n = 0
    total_ref_ms = 0.0

    for i, (ref, quant) in enumerate(zip(baseline_actions, quant_actions)):
        ref_f   = ref.float()
        quant_f = quant.float()
        diff    = ref_f - quant_f
        se      = diff.pow(2).sum().item()
        n       = diff.numel()
        ref_ms  = ref_f.pow(2).mean().item()
        rmse_i  = math.sqrt(se / n)
        ref_rms = math.sqrt(max(ref_ms, 0.0))
        rows.append({
            "obs_idx": i,
            "rmse":    rmse_i,
            "ref_rms": ref_rms,
            "rel_rmse": rmse_i / ref_rms if ref_rms > 0 else float("nan"),
        })
        total_se  += se
        total_n   += n
        total_ref_ms += ref_ms

    overall_rmse = math.sqrt(total_se / total_n) if total_n > 0 else float("nan")
    overall_ref_rms = math.sqrt(max(total_ref_ms / len(baseline_actions), 0.0))
    rows.append({
        "obs_idx": "overall",
        "rmse":    overall_rmse,
        "ref_rms": overall_ref_rms,
        "rel_rmse": overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan"),
    })

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_ACTION_RMSE_FIELDS)
        w.writeheader()
        w.writerows(rows)

    return overall_rmse, overall_ref_rms


_SWEEP_SUMMARY_FIELDS = ["extra_bits", "overall_rmse", "rel_rmse", "elapsed_s"]


def _write_sweep_summary(path: Path, rows: list[dict]) -> None:
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_SWEEP_SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


_OUTPUT_SUMMARY_FIELDS = [
    "timestamp", "label", "overall_rmse", "rel_rmse", "elapsed_seconds", "elapsed_human",
    "mx_input", "mx_output", "vec_input", "vec_output", "functional_model", "active_groups",
    "n_obs", "steps", "command",
]


def _append_output_summary(
    results_dir: Path,
    label: str,
    extra_bits: int,
    overall_rmse: float,
    overall_ref_rms: float,
    elapsed_s: float,
    mx_input_fmt: Optional[str],
    mx_output_fmt: Optional[str],
    vec_input_fmt: Optional[str],
    vec_output_fmt: Optional[str],
    active_groups: set[QuantGroup],
    n_obs: int,
    steps: int,
    command: str,
) -> None:
    path = results_dir / "sweep_extra_bits" / "all_runs_output_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    elapsed_td = str(datetime.timedelta(seconds=int(elapsed_s))) if math.isfinite(elapsed_s) else ""
    rel_rmse = overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan")
    row = {
        "timestamp":        datetime.datetime.now().isoformat(timespec="seconds"),
        "label":            f"{label}_eb{extra_bits}",
        "overall_rmse":     overall_rmse,
        "rel_rmse":         rel_rmse,
        "elapsed_seconds":  round(elapsed_s, 2),
        "elapsed_human":    elapsed_td,
        "mx_input":         mx_input_fmt or "passthrough",
        "mx_output":        mx_output_fmt or "passthrough",
        "vec_input":        vec_input_fmt or "passthrough",
        "vec_output":       vec_output_fmt or "passthrough",
        "functional_model": f"ipt_numba_exp(extra_bits={extra_bits})",
        "active_groups":    "|".join(g.value for g in active_groups),
        "n_obs":            n_obs,
        "steps":            steps,
        "command":          command,
    }
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_OUTPUT_SUMMARY_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep extra_bits for ipt_numba_exp and measure action output RMSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--label", required=True,
                        help="Run label — used as the output folder name under results-dir")

    parser.add_argument("--checkpoint-dir", default="/scratch/chloe.wong/data/pi05_base")
    parser.add_argument("--config", default="pi05_droid_jointpos_polaris")
    parser.add_argument("--gpu",    type=int, default=0)

    parser.add_argument("--n-obs",  type=int, default=1)
    parser.add_argument("--steps",  type=int, default=10)
    parser.add_argument("--seed",   type=int, default=0)

    parser.add_argument("--extra-bits-min",  type=int, default=0,
                        help="Minimum extra_bits value (inclusive, default: 0)")
    parser.add_argument("--extra-bits-max",  type=int, default=17,
                        help="Maximum extra_bits value (inclusive, default: 17)")
    parser.add_argument("--extra-bits-step", type=int, default=1,
                        help="Step size for extra_bits sweep (default: 1)")

    parser.add_argument("--mx-input-fmt",  metavar="FMT", default=PASSTHROUGH,
                        help="Format for matmul inputs alongside ipt_numba_exp. Use 'passthrough' for no-op.")
    parser.add_argument("--mx-output-fmt", metavar="FMT", default=PASSTHROUGH,
                        help="Format for matmul outputs (no effect with functional model).")

    parser.add_argument("--vec-input-fmt",  metavar="FMT", default=PASSTHROUGH)
    parser.add_argument("--vec-output-fmt", metavar="FMT", default=PASSTHROUGH)

    all_group_names = [g.value for g in QuantGroup]
    parser.add_argument("--active-groups", metavar="G1,G2,...",
                        default=",".join(all_group_names),
                        help=f"Comma-separated groups to quantize. Choices: {all_group_names}")

    parser.add_argument("--results-dir",
                        default=str(_REPO / "experiments" / "results"))
    parser.add_argument("--fp8-mode", default="po2", choices=["po2", "abs"])
    parser.add_argument("--norm-stats-dir", metavar="DIR", default=None)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-quantile-norm", action="store_true")

    args = parser.parse_args()

    # ── Validate ─────────────────────────────────────────────────────────────
    if args.extra_bits_min < 0:
        parser.error("--extra-bits-min must be >= 0")
    if args.extra_bits_max < args.extra_bits_min:
        parser.error("--extra-bits-max must be >= --extra-bits-min")
    if args.extra_bits_step < 1:
        parser.error("--extra-bits-step must be >= 1")

    active_groups: set[QuantGroup] = set()
    for g in args.active_groups.split(","):
        g = g.strip()
        try:
            active_groups.add(QuantGroup(g))
        except ValueError:
            parser.error(f"Unknown group '{g}'. Choices: {all_group_names}")

    mx_input_fmt   = _fmt_or_passthrough(args.mx_input_fmt)
    mx_output_fmt  = _fmt_or_passthrough(args.mx_output_fmt)
    vec_input_fmt  = _fmt_or_passthrough(args.vec_input_fmt)
    vec_output_fmt = _fmt_or_passthrough(args.vec_output_fmt)

    set_fp8_mode(args.fp8_mode)

    extra_bits_values = list(range(args.extra_bits_min, args.extra_bits_max + 1, args.extra_bits_step))
    print(f"extra_bits sweep: {extra_bits_values}")

    # ── Device / model ────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    torch.manual_seed(args.seed)
    print(f"Diffusion noise seed: {args.seed} (torch.manual_seed — fixed for reproducibility)")

    from pi0_inout.serve_quant import load_pi0_pytorch, _get_model_config
    _get_model_config(args.config)
    print(f"Loading model: {args.config}  checkpoint: {args.checkpoint_dir}")
    model = load_pi0_pytorch(args.config, args.checkpoint_dir, device)
    model.eval()

    # ── Norm stats ────────────────────────────────────────────────────────────
    norm_stats = None
    if args.norm_stats_dir:
        norm_stats = _load_norm_stats(args.norm_stats_dir)
        print(f"Loaded norm stats from {args.norm_stats_dir}")
    else:
        print("WARNING: no norm stats found — actions will be in normalized space.", file=sys.stderr)

    is_joint_position = "jointpos" in args.config
    policy = Pi0PyTorchPolicy(
        model=model,
        device=device,
        norm_stats=norm_stats,
        use_quantile_norm=args.use_quantile_norm,
        is_joint_position=is_joint_position,
        max_token_len=_get_model_config(args.config).max_token_len,
        tokenizer_path=args.tokenizer_path,
    )

    rng = np.random.default_rng(args.seed)
    observations = [_make_dummy_obs(rng) for _ in range(args.n_obs)]
    print(f"Observations: {args.n_obs}  steps: {args.steps}  seed: {args.seed}")

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_name = f"{run_ts}_ipt_numba_exp_steps{args.steps}"
    out_dir = Path(args.results_dir) / "sweep_extra_bits" / run_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n")

    # ── Baseline pass (once) ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("=== Baseline (unpatched) pass — runs once for all extra_bits ===")
    t0 = time.monotonic()
    torch.manual_seed(args.seed)
    baseline_actions = _collect_actions(policy, observations, args.steps, "baseline", t0)
    baseline_elapsed = time.monotonic() - t0
    print(f"Baseline done in {baseline_elapsed:.1f}s")
    _arr = np.stack([a.numpy() for a in baseline_actions])
    (out_dir / "baseline_actions.txt").write_text(f"shape: {_arr.shape}\ndtype: {_arr.dtype}\n{_arr}\n")

    # ── Register ipt_numba_exp variants and sweep ─────────────────────────────
    sweep_rows: list[dict] = []

    for extra_bits in extra_bits_values:
        factory_name = f"ipt_numba_exp_eb{extra_bits}"
        register_functional_model(
            factory_name,
            lambda in_f, out_f, eb=extra_bits: IPTNumbaExpLinear(extra_bits=eb),
        )

        print(f"\n{'='*60}")
        print(f"=== extra_bits={extra_bits} ({extra_bits_values.index(extra_bits)+1}/{len(extra_bits_values)}) ===")
        eb_t0 = time.monotonic()
        torch.manual_seed(args.seed)

        quant_actions = _run_quantized_pass(
            policy=policy,
            observations=observations,
            active_groups=active_groups,
            mx_input_fmt=mx_input_fmt,
            mx_output_fmt=mx_output_fmt,
            vec_input_fmt=vec_input_fmt,
            vec_output_fmt=vec_output_fmt,
            functional_model_name=factory_name,
            num_steps=args.steps,
            t0=eb_t0,
        )

        elapsed_s = time.monotonic() - eb_t0

        _arr = np.stack([a.numpy() for a in quant_actions])
        (out_dir / f"quant_actions_eb{extra_bits}.txt").write_text(f"shape: {_arr.shape}\ndtype: {_arr.dtype}\n{_arr}\n")

        overall_rmse, overall_ref_rms = _write_action_rmse(
            out_dir / f"action_rmse_eb{extra_bits}.csv",
            baseline_actions,
            quant_actions,
        )

        rel_rmse = overall_rmse / overall_ref_rms if overall_ref_rms > 0 else float("nan")

        sweep_rows.append({
            "extra_bits": extra_bits,
            "overall_rmse": overall_rmse,
            "rel_rmse": rel_rmse,
            "elapsed_s": round(elapsed_s, 2),
        })

        _append_output_summary(
            Path(args.results_dir),
            label=args.label,
            extra_bits=extra_bits,
            overall_rmse=overall_rmse,
            overall_ref_rms=overall_ref_rms,
            elapsed_s=elapsed_s,
            mx_input_fmt=args.mx_input_fmt if args.mx_input_fmt != PASSTHROUGH else None,
            mx_output_fmt=args.mx_output_fmt if args.mx_output_fmt != PASSTHROUGH else None,
            vec_input_fmt=args.vec_input_fmt if args.vec_input_fmt != PASSTHROUGH else None,
            vec_output_fmt=args.vec_output_fmt if args.vec_output_fmt != PASSTHROUGH else None,
            active_groups=active_groups,
            n_obs=args.n_obs,
            steps=args.steps,
            command=" ".join(sys.argv),
        )

        print(f"  extra_bits={extra_bits:>3}  rmse={overall_rmse:.4e}  rel_rmse={rel_rmse:.4e}  t={elapsed_s:.1f}s")

    # ── Write sweep summary CSV ───────────────────────────────────────────────
    sweep_csv = out_dir / "sweep_summary.csv"
    _write_sweep_summary(sweep_csv, sweep_rows)

    # ── Print final table ─────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"Sweep complete  total={datetime.timedelta(seconds=int(total_elapsed))}")
    print(f"Results dir:    {out_dir}")
    print(f"Sweep summary:  {sweep_csv}")
    print(f"Top-level:      {Path(args.results_dir) / 'sweep_extra_bits' / 'all_runs_output_summary.csv'}")
    print(f"\n  {'extra_bits':>10}  {'overall_rmse':>14}  {'rel_rmse':>12}  {'elapsed_s':>10}")
    print(f"  {'-'*10}  {'-'*14}  {'-'*12}  {'-'*10}")
    for row in sweep_rows:
        print(
            f"  {row['extra_bits']:>10}  {row['overall_rmse']:>14.6e}"
            f"  {row['rel_rmse']:>12.6e}  {row['elapsed_s']:>10.1f}"
        )


if __name__ == "__main__":
    main()
