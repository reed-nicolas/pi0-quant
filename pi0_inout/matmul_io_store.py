"""
matmul_io_store.py
------------------
Captures per-layer matmul inputs and outputs for both the unpatched reference
pass and the patched (quantized) pass, then writes one .npz file per layer.

Captured tensors
----------------
Unpatched pass (via forward hooks registered in run_eval.py):
    unpatched_x       [N, *]  activation arriving at this layer
    unpatched_w       [out, in]  weight (stored once — static)
    unpatched_b       [out]  bias (stored once; absent if no bias)
    unpatched_y       [N, *]  F.linear(x, w, b) output

Patched pass (via QuantLinear.forward):
    patched_x         [N, *]  activation in patched forward (may be corrupted by earlier layers)
    patched_x_q       [N, *]  quantized activation (format-flag path only; absent for functional model)
    patched_w_q       [out, in]  quantized weight (format-flag; stored once)
    patched_b_q       [out]  quantized bias (format-flag; absent if no bias or functional model)
    patched_y_quant   [N, *]  quantized output

All arrays stored as float32 in the .npz (numpy does not support bfloat16).

File naming
-----------
Layer names use dots as separators (e.g.
    paligemma_with_expert.paligemma.language_model.model.layers.0.self_attn.q_proj
).  Dots are replaced with '__' so the name is a valid filename:
    paligemma_with_expert__paligemma__language_model__model__layers__0__self_attn__q_proj.npz
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_f32_numpy(t: torch.Tensor) -> np.ndarray:
    """CPU transfer + float32 cast + numpy conversion."""
    return t.detach().to(torch.float32).cpu().numpy()


def _maybe_f32_numpy(t: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    return None if t is None else _to_f32_numpy(t)


def _stack_calls(
    calls: List[dict],
    prefix: str,
    static_keys: set,
) -> dict:
    """
    Stack per-call dicts of numpy arrays.

    - Dynamic keys (activations, outputs): stack along new axis 0 → [N, ...]
    - Static keys (weight, bias): take from the first call that has them
    - Keys with all-None values across calls are skipped entirely

    Falls back to per-call keys (``prefix_key_callN``) if shapes are
    inconsistent across calls.
    """
    all_keys = {k for c in calls for k in c if c[k] is not None}
    out: dict = {}

    for key in sorted(all_keys):
        prefixed = f"{prefix}_{key}"
        if key in static_keys:
            for c in calls:
                if c.get(key) is not None:
                    out[prefixed] = c[key]
                    break
        else:
            frames = [c[key] for c in calls if c.get(key) is not None]
            if not frames:
                continue
            try:
                out[prefixed] = np.stack(frames, axis=0)
            except ValueError:
                # Shape mismatch across calls — fall back to per-call keys
                for i, arr in enumerate(frames):
                    out[f"{prefixed}_call{i}"] = arr

    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MatmulIOStore:
    """
    Accumulates per-layer matmul I/O tensors across inference calls and saves
    them as one .npz file per layer.

    Usage in run_eval.py:
        store = MatmulIOStore(out_dir / "tensors")
        # Register unpatched hooks before reference pass (see run_eval.py)
        # Pass store to patch_model(..., matmul_io_store=store)
        # Run inference
        store.save()
    """

    def __init__(self, save_dir: Path) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # layer_name -> list of per-call dicts
        self._unpatched: Dict[str, List[dict]] = defaultdict(list)
        self._patched:   Dict[str, List[dict]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @torch._dynamo.disable
    def record_unpatched(
        self,
        name: str,
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        y: torch.Tensor,
    ) -> None:
        """Called from the forward hook on nn.Linear during the reference pass."""
        self._unpatched[name].append({
            "x": _to_f32_numpy(x),
            "w": _to_f32_numpy(w),
            "b": _maybe_f32_numpy(b),
            "y": _to_f32_numpy(y),
        })

    @torch._dynamo.disable
    def record_patched(
        self,
        name: str,
        x: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor],
        x_q: Optional[torch.Tensor],
        w_q: Optional[torch.Tensor],
        b_q: Optional[torch.Tensor],
        y_quant: torch.Tensor,
    ) -> None:
        """Called from QuantLinear.forward() during the patched pass."""
        self._patched[name].append({
            "x":       _to_f32_numpy(x),
            "w":       _to_f32_numpy(w),
            "b":       _maybe_f32_numpy(b),
            "x_q":     _maybe_f32_numpy(x_q),
            "w_q":     _maybe_f32_numpy(w_q),
            "b_q":     _maybe_f32_numpy(b_q),
            "y_quant": _to_f32_numpy(y_quant),
        })

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write one .npz per layer. Call after all inference is complete."""
        all_names = sorted(set(self._unpatched) | set(self._patched))
        if not all_names:
            print("[MatmulIOStore] No tensors recorded — nothing to save.")
            return

        for name in all_names:
            arrays: dict = {}
            up_calls  = self._unpatched.get(name, [])
            pat_calls = self._patched.get(name, [])
            n_calls   = max(len(up_calls), len(pat_calls))
            arrays["n_calls"] = np.array(n_calls, dtype=np.int64)

            if up_calls:
                arrays.update(
                    _stack_calls(up_calls, prefix="unpatched", static_keys={"w", "b"})
                )
            if pat_calls:
                arrays.update(
                    _stack_calls(pat_calls, prefix="patched", static_keys={"w", "b", "w_q", "b_q"})
                )

            fname = name.replace(".", "__") + ".npz"
            np.savez(self.save_dir / fname, **arrays)

        print(
            f"[MatmulIOStore] Saved {len(all_names)} layer .npz files → {self.save_dir}"
        )
