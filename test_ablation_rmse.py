"""
test_ablation_rmse.py
---------------------
Ablation RMSE sweep over all 7 subsets of {vision, transformer, action_head}.

Metrics:
  act_rmse            — action output RMSE vs bf16 baseline (absolute,
                        normalized joint space ≈ [-1, 1])
  act_snr_db          — SNR in dB: 20*log10(rms_baseline / act_rmse)
  max_act_err         — worst-case single action element error (safety metric)
  act_cos_sim         — mean cosine similarity of action vectors vs baseline
  avg_rel_rmse        — mean(rmse / rms_fp32) across all quantized linear layers
  pct_layers_above_5  — % of quantized layers with rel_rmse > 5%
  rel_vision / rel_language / rel_act_exp / rel_act_head
                      — per-component mean relative RMSE
  joint_{0..7}_rmse   — per-DOF action RMSE (joints 0-6 + gripper)

Data: real DROID frames from droid_100 (parquet + MP4 videos).
      Images decoded and letterbox-resized to 224×224.

Usage:
    OPENPI_DIR=/scratch/chloe.wong/openpi \
    CUDA_VISIBLE_DEVICES=0 \
    /scratch/chloe.wong/envs/pi0/bin/python test_ablation_rmse.py

Results saved to ablation_results.csv in the same directory.
"""

from __future__ import annotations

import csv
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_DIR   = Path(__file__).resolve().parent
_OPENPI_DIR = Path(os.environ.get("OPENPI_DIR", _THIS_DIR / "openpi"))
_OPENPI_SRC = _OPENPI_DIR / "src"
_CLIENT_SRC = _OPENPI_DIR / "packages" / "openpi-client" / "src"

for _p in [str(_THIS_DIR), str(_CLIENT_SRC), str(_OPENPI_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pi0_inout._jax_stubs import inject as _inject_jax_stubs
_inject_jax_stubs()

from pi0_inout import (
    patch_model, unpatch_model,
    patch_attn_sdpa, unpatch_attn_sdpa,
    QuantFormat, QuantGroup, StatsTracker,
)
from pi0_inout.serve_quant import load_pi0_pytorch

import logging
logging.basicConfig(level=logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "/scratch/chloe.wong/data/pi05_base"
CONFIG_NAME    = "pi05_droid_jointpos_polaris"
DROID_ROOT     = Path("/scratch/chloe.wong/data/droid_100")
DEVICE         = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

INPUT_FMT  = QuantFormat.FLOAT8_E4M3
OUTPUT_FMT = QuantFormat.FLOAT16

N_FRAMES   = 8   # real frames to evaluate on
N_JOINTS   = 8   # 7 arm joints + 1 gripper
REL_THRESH = 0.05  # layers above this rel_rmse are flagged (5%)

CSV_OUTPUT = Path(__file__).resolve().parent / "ablation_results_4group.csv"

ALL_GROUPS = [
    QuantGroup.VISION,
    QuantGroup.LANGUAGE,
    QuantGroup.ACTION_EXPERT,
    QuantGroup.ACTION_HEAD,
]

# All non-empty subsets of the 4 groups (15 total: 4C1+4C2+4C3+4C4)
CONFIGS: list[tuple[str, set[QuantGroup]]] = []
for r in [1, 2, 3, 4]:
    for combo in combinations(ALL_GROUPS, r):
        label = "+".join(g.value for g in combo)
        CONFIGS.append((label, set(combo)))

VIDEO_KEYS = {
    "base_0_rgb":       "observation.images.exterior_image_1_left",
    "left_wrist_0_rgb": "observation.images.wrist_image_left",
    "ext2_rgb":         "observation.images.exterior_image_2_left",
}
ACTION_DIM   = 32
MAX_TOK      = 200


# ── Data loading ──────────────────────────────────────────────────────────────

def _letterbox_resize(frame_hwc: np.ndarray, target: int = 224) -> np.ndarray:
    """
    Letterbox-resize a uint8 HWC image to (target, target).
    Pads with zeros (black).  Mirrors the openpi resize_with_pad convention.
    """
    h, w = frame_hwc.shape[:2]
    scale = min(target / h, target / w)
    new_h, new_w = int(h * scale), int(w * scale)

    t = torch.from_numpy(frame_hwc).permute(2, 0, 1).unsqueeze(0).float()
    t = F.interpolate(t, size=(new_h, new_w), mode="bilinear", align_corners=False)
    t = torch.round(t).clamp(0, 255).to(torch.uint8)

    out = torch.zeros(1, 3, target, target, dtype=torch.uint8)
    pad_h = (target - new_h) // 2
    pad_w = (target - new_w) // 2
    out[:, :, pad_h:pad_h + new_h, pad_w:pad_w + new_w] = t
    return out.squeeze(0).permute(1, 2, 0).numpy()   # HWC uint8


def _load_videos() -> dict[str, list[np.ndarray]]:
    """
    Decode all frames from each camera MP4 into a list indexed by global frame index.
    Returns {camera_key: [frame_hwc, ...]} with len == 32212.
    """
    videos: dict[str, list[np.ndarray]] = {}
    for key, folder in VIDEO_KEYS.items():
        mp4_path = DROID_ROOT / "videos" / folder / "chunk-000" / "file-000.mp4"
        frames = []
        container = av.open(str(mp4_path))
        for packet in container.decode(video=0):
            frames.append(packet.to_ndarray(format="rgb24"))
        container.close()
        videos[key] = frames
        print(f"  Loaded {len(frames)} frames from {folder.split('.')[-1]}")
    return videos


def _select_frame_indices(df: pd.DataFrame, n: int) -> list[int]:
    """
    Pick n frames evenly spread across different episodes so we sample
    diverse scenes rather than a burst from one episode.
    """
    episodes = sorted(df["episode_index"].unique())
    step = max(1, len(episodes) // n)
    chosen_eps = episodes[::step][:n]
    indices = []
    for ep in chosen_eps:
        ep_rows = df[df["episode_index"] == ep].sort_values("frame_index")
        # Take the midpoint frame of each episode
        mid = ep_rows.iloc[len(ep_rows) // 2]
        indices.append(int(mid["index"]))
    return indices


def _img_tensor(frame_hwc: np.ndarray, device: torch.device) -> torch.Tensor:
    """uint8 HWC 224×224 → float [1, 3, H, W] in [-1, 1]."""
    t = torch.from_numpy(frame_hwc).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t / 255.0 * 2.0 - 1.0


def build_observations(
    df: pd.DataFrame,
    videos: dict[str, list[np.ndarray]],
    frame_indices: list[int],
    tasks_df: pd.DataFrame,
    device: torch.device,
) -> list[SimpleNamespace]:
    obs_list = []
    for idx in frame_indices:
        row = df[df["index"] == idx].iloc[0]

        # Images — letterbox resize to 224×224 then normalise
        base_img  = _letterbox_resize(videos["base_0_rgb"][idx])
        wrist_img = _letterbox_resize(videos["left_wrist_0_rgb"][idx])

        images = {
            "base_0_rgb":        _img_tensor(base_img,  device),
            "left_wrist_0_rgb":  _img_tensor(wrist_img, device),
            "right_wrist_0_rgb": _img_tensor(base_img,  device),  # duplicate; masked out
        }
        image_masks = {
            "base_0_rgb":        torch.ones(1,  dtype=torch.bool, device=device),
            "left_wrist_0_rgb":  torch.ones(1,  dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool, device=device),
        }

        # State: 7-dim → pad to ACTION_DIM=32
        state_raw = np.array(row["observation.state"], dtype=np.float32)
        state_pad = np.zeros(ACTION_DIM, dtype=np.float32)
        state_pad[:len(state_raw)] = state_raw
        state = torch.from_numpy(state_pad).unsqueeze(0).to(device)

        # Prompt: zero tokens (we don't have a tokenizer here; model still runs)
        tokenized_prompt      = torch.zeros(1, MAX_TOK, dtype=torch.int64, device=device)
        tokenized_prompt_mask = torch.zeros(1, MAX_TOK, dtype=torch.bool,  device=device)
        token_ar_mask         = torch.zeros(1, MAX_TOK, dtype=torch.bool,  device=device)
        token_loss_mask       = torch.zeros(1, MAX_TOK, dtype=torch.bool,  device=device)

        obs_list.append(SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
        ))
    return obs_list


# ── Evaluation ────────────────────────────────────────────────────────────────

SEEDS = [42 + i for i in range(N_FRAMES)]  # fixed per-frame seeds


def run_baseline(model, observations) -> list[torch.Tensor]:
    results = []
    with torch.no_grad():
        for seed, obs in zip(SEEDS, observations):
            torch.manual_seed(seed)
            actions = model.sample_actions(str(DEVICE), obs, num_steps=10)
            results.append(actions.detach().cpu())
    return results


def run_config(
    model,
    observations: list,
    baseline_actions: list[torch.Tensor],
    active_groups: set[QuantGroup],
) -> dict:
    tracker = StatsTracker()
    patch_model(model, input_fmt=INPUT_FMT, output_fmt=OUTPUT_FMT,
                tracker=tracker, active_groups=active_groups)
    attn_handles = patch_attn_sdpa(model, active_groups=active_groups,
                                   input_fmt=INPUT_FMT, output_fmt=OUTPUT_FMT,
                                   tracker=tracker)

    quant_actions = []
    with torch.no_grad():
        for seed, obs in zip(SEEDS, observations):
            torch.manual_seed(seed)   # same seed as baseline → isolates quant error
            actions = model.sample_actions(str(DEVICE), obs, num_steps=10)
            quant_actions.append(actions.detach().cpu())

    unpatch_attn_sdpa(attn_handles)
    unpatch_model(model)

    # Per-frame error tensors — shape (num_steps, n_joints) each
    errors = [ref.float() - q.float() for ref, q in zip(baseline_actions, quant_actions)]

    # Action output RMSE (absolute, averaged over frames)
    act_rmse = float(np.mean([e.pow(2).mean().sqrt().item() for e in errors]))

    # SNR in dB: 20*log10(rms_signal / rmse)
    rms_baseline = float(np.mean([
        ref.float().pow(2).mean().sqrt().item() for ref in baseline_actions
    ]))
    act_snr_db = (20.0 * math.log10(rms_baseline / act_rmse)
                  if act_rmse > 0 else float("inf"))

    # Worst-case single action element error across all frames
    max_act_err = float(max(e.abs().max().item() for e in errors))

    # Cosine similarity: flatten (num_steps, n_joints) → 1D vector per frame
    cos_sims = [
        F.cosine_similarity(ref.float().flatten().unsqueeze(0),
                            q.float().flatten().unsqueeze(0)).item()
        for ref, q in zip(baseline_actions, quant_actions)
    ]
    act_cos_sim = float(np.mean(cos_sims))

    # Per-joint RMSE: mean over time steps, then average over frames
    joint_rmse_per_frame = [e.pow(2).mean(dim=0).sqrt() for e in errors]  # each (n_joints,)
    per_joint_rmse = torch.stack(joint_rmse_per_frame).mean(dim=0).tolist()
    # Pad or trim to N_JOINTS in case action dim differs
    per_joint_rmse = (per_joint_rmse + [float("nan")] * N_JOINTS)[:N_JOINTS]

    # Average relative RMSE across all quantized layers
    report = tracker.summary()
    all_rel = [
        row["rel_rmse"]
        for row in report.to_dict().get("layers", [])
        if row["rel_rmse"] is not None and not math.isnan(row["rel_rmse"])
    ]
    avg_rel_rmse = float(np.mean(all_rel)) if all_rel else float("nan")

    # % of layers with rel_rmse above threshold
    n_above = sum(1 for v in all_rel if v > REL_THRESH)
    pct_layers_above_5 = 100.0 * n_above / len(all_rel) if all_rel else float("nan")

    # Per-component breakdown
    comp_rel = {
        row["component"]: row["mean_rel_rmse"]
        for row in report.to_dict().get("components", [])
    }

    return {
        "act_rmse":           act_rmse,
        "act_snr_db":         act_snr_db,
        "max_act_err":        max_act_err,
        "act_cos_sim":        act_cos_sim,
        "avg_rel_rmse":       avg_rel_rmse,
        "pct_layers_above_5": pct_layers_above_5,
        "comp_rel":           comp_rel,
        "per_joint_rmse":     per_joint_rmse,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

def print_results(results: list[dict]) -> None:
    comp_cols    = ["vision", "language", "action_expert", "action_head"]
    comp_headers = ["rel_vision", "rel_language", "rel_act_exp", "rel_act_head"]
    cw = 13

    header = (f"{'config':<28s}  {'act_rmse':>{cw}}  {'snr_db':>{cw}}  "
              f"{'max_err':>{cw}}  {'cos_sim':>{cw}}  {'avg_rel_rmse':>{cw}}  "
              f"{'pct_>5%':>{cw}}")
    for h in comp_headers:
        header += f"  {h:>{cw}}"
    sep = "-" * len(header)

    print(f"\n{'=' * len(header)}")
    print(f"ABLATION RMSE  (input={INPUT_FMT.value}  output={OUTPUT_FMT.value}  "
          f"n_frames={N_FRAMES})")
    print(f"{'=' * len(header)}")
    print(header)
    print(sep)

    for r in results:
        row = (f"{r['label']:<28s}  {r['act_rmse']:>{cw}.4e}  "
               f"{r['act_snr_db']:>{cw}.2f}  "
               f"{r['max_act_err']:>{cw}.4e}  "
               f"{r['act_cos_sim']:>{cw}.6f}  "
               f"{r['avg_rel_rmse']:>{cw}.4e}  "
               f"{r['pct_layers_above_5']:>{cw}.1f}")
        for c in comp_cols:
            v = r["comp_rel"].get(c)
            row += f"  {(f'{v:.4e}' if v is not None else 'N/A'):>{cw}}"
        print(row)

    print(f"{'=' * len(header)}")
    print("act_rmse     = action output RMSE vs bf16 baseline (normalized joint space)")
    print("act_snr_db   = 20*log10(rms_baseline / act_rmse)")
    print("max_act_err  = worst-case single action element error")
    print("cos_sim      = mean cosine similarity of action vectors vs baseline")
    print("avg_rel_rmse = mean(rmse / rms_fp32) across all quantized linear layers")
    print("pct_>5%      = % of quantized layers with rel_rmse > 5%")
    print("rel_*        = per-component mean relative RMSE")


def save_csv(results: list[dict], path: Path) -> None:
    comp_cols = ["vision", "language", "action_expert", "action_head"]
    fieldnames = (
        ["config", "act_rmse", "act_snr_db", "max_act_err", "act_cos_sim",
         "avg_rel_rmse", "pct_layers_above_5"]
        + [f"rel_{c}" for c in comp_cols]
        + [f"joint_{i}_rmse" for i in range(N_JOINTS)]
    )
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row: dict = {
                "config":             r["label"],
                "act_rmse":           r["act_rmse"],
                "act_snr_db":         r["act_snr_db"],
                "max_act_err":        r["max_act_err"],
                "act_cos_sim":        r["act_cos_sim"],
                "avg_rel_rmse":       r["avg_rel_rmse"],
                "pct_layers_above_5": r["pct_layers_above_5"],
            }
            for c in comp_cols:
                row[f"rel_{c}"] = r["comp_rel"].get(c, float("nan"))
            for i, v in enumerate(r["per_joint_rmse"]):
                row[f"joint_{i}_rmse"] = v
            writer.writerow(row)
    print(f"\nResults saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print(f"Loading model ...")
    model = load_pi0_pytorch(CONFIG_NAME, CHECKPOINT_DIR, DEVICE)
    print(f"Model loaded on {DEVICE}.\n")

    print(f"Loading DROID videos (this takes ~30s) ...")
    videos = _load_videos()

    print(f"\nSelecting {N_FRAMES} real frames ...")
    df = pd.read_parquet(DROID_ROOT / "data" / "chunk-000" / "file-000.parquet")
    tasks_df = pd.read_parquet(DROID_ROOT / "meta" / "tasks.parquet")
    frame_indices = _select_frame_indices(df, N_FRAMES)
    print(f"  Frame indices: {frame_indices}")

    print(f"\nBuilding observations ...")
    observations = build_observations(df, videos, frame_indices, tasks_df, DEVICE)

    print(f"\nRunning bf16 baseline ({N_FRAMES} frames) ...")
    baseline_actions = run_baseline(model, observations)
    print("Baseline done.\n")

    all_results = []
    for label, active_groups in CONFIGS:
        print(f"Config: {label}")
        r = run_config(model, observations, baseline_actions, active_groups)
        r["label"] = label
        all_results.append(r)
        print(f"  act_rmse={r['act_rmse']:.4e}  avg_rel_rmse={r['avg_rel_rmse']:.4e}")

    print_results(all_results)
    save_csv(all_results, CSV_OUTPUT)


if __name__ == "__main__":
    main()
