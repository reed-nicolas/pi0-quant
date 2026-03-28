# pi0-quant

Matmul and vector-op quantization framework for [Pi0Pytorch](https://github.com/Physical-Intelligence/openpi).

Replaces every `nn.Linear` in the model with a `QuantLinear` that simulates
reduced-precision matmuls, intercepts attention score matmuls, and optionally
quantizes all vector operations (layernorm, activations, elementwise ops).
Supports both software format-flag quantization and hardware-accurate functional
models (e.g. Inner Product Tree simulation).

## What it quantizes

### Matrix path (linear layers + attention)

Every `nn.Linear` is replaced with `QuantLinear`:

```
x_q  = quant(x, mx_input_fmt)    # activation snapped to input format
W_q  = quant(W, mx_input_fmt)    # weight snapped to input format
b_q  = quant(b, mx_input_fmt)    # bias snapped to input format
y    = x_q @ W_q^T + b_q         # matmul in original model dtype
out  = quant(y, mx_output_fmt)   # output snapped to output format
```

Attention score matmuls (`Q@K^T` and `weights@V`) are handled separately
via `patch_attn_sdpa`, which monkey-patches `F.scaled_dot_product_attention`.

Alternatively, a **functional model** can replace the matmul entirely with a
hardware-accurate simulation (see [Functional models](#functional-models) below).

### Vector path

`patch_vector_ops` installs a `TorchDispatchMode` that intercepts all
vector operations in the model:

`add`, `sub`, `mul`, `pow`, `div`, `reciprocal`, `sqrt`, `sin`, `cos`,
`tanh`, `log2`, `exp`, `exp2`, `amax`, `sum`

Each intercepted op has its inputs quantized to `vec_input_fmt` and its
output quantized to `vec_output_fmt`.

## Architecture components

RMSE is measured per layer and aggregated across four components:

| Component | Layers |
|---|---|
| `vision` | SigLIP ViT encoder (inside PaliGemma) |
| `language` | Gemma 2.6B language model (inside PaliGemma) |
| `action_expert` | Gemma 300M action-expert transformer |
| `action_head` | Action projection MLPs at the Pi0 root |

## Requirements

- Python ≥ 3.10, PyTorch ≥ 2.1
- [openpi](https://github.com/Physical-Intelligence/openpi) on your Python path
- A Pi0 checkpoint in safetensors format
- `numba` if using `ipt_numba`: `pip install numba`

## Quick start

### Run the evaluation script

```bash
OPENPI_DIR=/path/to/openpi \
CUDA_VISIBLE_DEVICES=0 \
python experiments/run_eval.py \
    --label fp8_mx_only \
    --mx-input-fmt float8_e4m3 --mx-output-fmt bfloat16 \
    --checkpoint-dir /path/to/pi05_base \
    --config pi05_droid_jointpos_polaris
```

Results are written to `experiments/results/<label>/`:
- `config.json` — exact parameters used
- `chronological.csv` — one row per op call in execution order
- `grouped.csv` — same rows sorted by (component, layer_name)
- `summary.csv` — per-component aggregate RMSE stats

A top-level `experiments/results/all_runs_summary.csv` accumulates one row
per run across all configs.

### Common configs

```bash
# FP8 matmul inputs, FP16 outputs
python experiments/run_eval.py --label fp8_mx \
    --mx-input-fmt float8_e4m3 --mx-output-fmt float16

# FP8 matmul + FP8 vector ops
python experiments/run_eval.py --label fp8_mx_vec \
    --mx-input-fmt float8_e4m3 --mx-output-fmt float16 \
    --vec-input-fmt float8_e4m3 --vec-output-fmt float16

# Hardware-accurate IPT simulation (C kernel, ~40 min)
python experiments/run_eval.py --label ipt_c \
    --functional-model ipt_c

# Quantize only action components
python experiments/run_eval.py --label fp8_action_only \
    --mx-input-fmt float8_e4m3 --mx-output-fmt float16 \
    --active-groups action_expert,action_head
```

### Key CLI flags

| Flag | Default | Description |
|---|---|---|
| `--label` | *(required)* | Output folder name under `results/` |
| `--mx-input-fmt` | passthrough | Format for matmul inputs |
| `--mx-output-fmt` | passthrough | Format for matmul outputs |
| `--functional-model` | — | Hardware sim instead of format flags (mutually exclusive with `--mx-input-fmt`) |
| `--vec-input-fmt` | passthrough | Format for vector op inputs |
| `--vec-output-fmt` | passthrough | Format for vector op outputs |
| `--active-groups` | all | Comma-separated components to quantize |
| `--n-obs` | 4 | Number of observations to run |
| `--steps` | 10 | Diffusion steps per observation |
| `--gpu` | 0 | CUDA device index |

## Functional models

Functional models replace the matmul inside each `QuantLinear` with a
hardware-accurate simulation. They receive raw `(x, w, b)` tensors and return
an accumulated result.

Three IPT (Inner Product Tree) variants are built in:

| Name | Description | Speed |
|---|---|---|
| `ipt` | Pure Python reference | Very slow (hours) |
| `ipt_numba` | Parallel Numba JIT kernel | Faster |
| `ipt_c` | C/ctypes compiled kernel | Fastest (~40 min) |

All three simulate: E4M3 inputs, BF16 partial sums at each accumulation step,
BF16 output — matching the hardware accumulation model.

### Adding a new functional model

```python
from pi0_inout.functional_models import register_functional_model

def my_factory(in_features: int, out_features: int):
    return MyModel(in_features, out_features)

register_functional_model("my_model", my_factory)
```

Then pass `--functional-model my_model` to `run_eval.py`. Each `QuantLinear`
gets its own instance (important for per-layer weight caching).

## Programmatic API

```python
from pi0_inout import (
    QuantFormat, QuantGroup,
    StatsTracker,
    patch_model, unpatch_model,
    patch_attn_sdpa, unpatch_attn_sdpa,
)
from pi0_inout.quant_vector import patch_vector_ops, unpatch_vector_ops

tracker = StatsTracker()
active  = {QuantGroup.LANGUAGE, QuantGroup.ACTION_EXPERT}

patch_model(model,
    mx_input_fmt=QuantFormat.FLOAT8_E4M3,
    mx_output_fmt=QuantFormat.BFLOAT16,
    tracker=tracker,
    active_groups=active,
)
attn_handles = patch_attn_sdpa(model,
    active_groups=active,
    mx_input_fmt=QuantFormat.FLOAT8_E4M3,
    mx_output_fmt=QuantFormat.BFLOAT16,
    tracker=tracker,
)
vec_handles, vec_ctx = patch_vector_ops(model,
    active_groups=active,
    vec_input_fmt=QuantFormat.FLOAT8_E4M3,
    vec_output_fmt=QuantFormat.BFLOAT16,
)

with torch.no_grad(), vec_ctx:
    actions = model.sample_actions(device, obs, num_steps=10)

tracker.summary().print()

unpatch_model(model)
unpatch_attn_sdpa(attn_handles)
unpatch_vector_ops(vec_handles)
```

### Using a functional model

```python
from pi0_inout.functional_models import get_functional_model_factory

factory = get_functional_model_factory("ipt_c")
patch_model(model,
    mx_input_fmt=QuantFormat.BFLOAT16,   # ignored when functional_model_factory set
    mx_output_fmt=QuantFormat.BFLOAT16,
    tracker=tracker,
    functional_model_factory=factory,
)
```

## Package layout

```
pi0_inout/
├── quant_types.py         # QuantFormat enum, quant()
├── quant_linear.py        # QuantLinear: drop-in nn.Linear replacement
├── quant_vector.py        # VectorQuantMode: TorchDispatchMode for vector ops
├── model_patcher.py       # patch_model(), unpatch_model(), patch_attn_sdpa()
├── functional_models.py   # Registry for functional model factories
├── stats_tracker.py       # StatsTracker: per-layer Welford RMSE accumulator
├── _dispatch_guards.py    # Shared re-entrancy guard (quant_linear + quant_vector)
├── eval_harness.py        # Lower-level eval utilities
├── serve_quant.py         # WebSocket server with quantized Pi0Pytorch
├── run_benchmark.py       # Full sweep orchestrator
└── _jax_stubs.py          # Stub modules so Pi0Pytorch loads without JAX

funct_models_ipt/
├── python_ipt_base/       # Pure Python IPT ("ipt")
├── ipt_numba/             # Numba JIT IPT ("ipt_numba")
└── ipt_c/                 # C/ctypes IPT ("ipt_c")

func_models_sa/            # Systolic array functional models (in progress)

experiments/
└── run_eval.py            # Main evaluation runner
```

### Why `_jax_stubs.py` exists

Several openpi source files import JAX at module level even though Pi0Pytorch
is pure PyTorch. `_jax_stubs.py` injects lightweight replacements into
`sys.modules` before those imports happen. `serve_quant.py` handles this
automatically; call `_jax_stubs.inject()` manually if loading Pi0Pytorch
directly.
