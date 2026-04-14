"""Targeted unit tests for the `TanhRec` lane box.

`TanhRec` mirrors `TanhBlock.scala` + `TanhLUT.scala` in
`sp26-fp-units/vpuFUnits/`. The RTL snapshot cross-test
(`test_rtl_actual_outputs.py`) catches drift from hardware across the
generated tanh cases; this file is the targeted corner coverage:
zero/sign preservation, the `|x|>=4` saturation branch (which the RTL
also uses to swallow NaN/inf), LUT address endpoints, and a mid-range
cross-check against `math.tanh`.
"""

from __future__ import annotations

import math
import struct

import pytest

from funct_models_vector.lane_boxes.tanh_rec import (
    TanhRec,
    TanhReq,
    tanh_bf16,
)
from funct_models_vector.lut_sources.lut_tables import gen_tanh_lut
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes

# Same LUT shape that `TanhRec.__init__` uses (mirrors `TanhRec.scala:52`).
LUT = gen_tanh_lut(addr_bits=5, m=1, n=16, minimum=0.0, maximum=4.0)


def _f32_to_bf16_bits(x: float) -> int:
    """Python float → BF16 bits with RNE, matching the model's NaN
    encoding. Inlined here so the test doesn't depend on a private
    helper inside `vector_engine_model.py`."""
    if math.isnan(x):
        return 0x7FC0
    if x == float("inf"):
        return 0x7F80
    if x == float("-inf"):
        return 0xFF80
    fp32 = struct.unpack(">I", struct.pack(">f", x))[0]
    lower = fp32 & 0xFFFF
    bit16 = (fp32 >> 16) & 1
    if lower > 0x8000 or (lower == 0x8000 and bit16 == 1):
        fp32 = (fp32 + 0x8000) & 0xFFFFFFFF
    return (fp32 >> 16) & 0xFFFF


def _bf16_bits_to_f32(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">I", (bits & 0xFFFF) << 16))[0]


# ----------------------------------------------------------------
#  Zero / sign preservation — `is_zero` forwarding in the recode tail.
# ----------------------------------------------------------------

def test_tanh_positive_zero_in_positive_zero_out():
    assert tanh_bf16(0x0000, LUT) == 0x0000


def test_tanh_negative_zero_preserves_sign():
    assert tanh_bf16(0x8000, LUT) == 0x8000


# ----------------------------------------------------------------
#  Saturation: |x| >= 4.0 → ±1.0 via the `is_saturated` branch.
# ----------------------------------------------------------------

@pytest.mark.parametrize("bits,expected", [
    (0x4080, 0x3F80),  # +4.0 → +1.0  (trueExp = 2, first saturated step)
    (0xC080, 0xBF80),  # -4.0 → -1.0
    (0x4100, 0x3F80),  # +8.0 → +1.0
    (0xC100, 0xBF80),  # -8.0 → -1.0
    (0x4880, 0x3F80),  # ~+1e4 → +1.0  (trueExp >> 2, well past sat)
    (0xC880, 0xBF80),  # ~-1e4 → -1.0
])
def test_tanh_saturation_branch(bits, expected):
    assert tanh_bf16(bits, LUT) == expected


# ----------------------------------------------------------------
#  Inf / NaN — caught by the same `is_saturated` branch (trueExp=128).
#  This is NOT IEEE `tanh(NaN) = NaN`; the Chisel module does not
#  special-case NaN, so both sides produce ±1.0 by construction.
# ----------------------------------------------------------------

@pytest.mark.parametrize("bits,expected", [
    (0x7F80, 0x3F80),  # +inf → +1.0
    (0xFF80, 0xBF80),  # -inf → -1.0
    (0x7FC0, 0x3F80),  # +NaN (quiet) → +1.0
    (0xFFC0, 0xBF80),  # -NaN (quiet, sign bit set) → -1.0
])
def test_tanh_inf_and_nan_map_through_saturation(bits, expected):
    assert tanh_bf16(bits, LUT) == expected


# ----------------------------------------------------------------
#  Inside the LUT interp path, with a result that is visibly below 1.0.
#  BF16(3.0) = 0x4040 has trueExp=1, so it is NOT saturated. We pick 3.0
#  specifically because tanh(3.0) ≈ 0.9951 rounds to BF16 0x3F7F —
#  visibly distinct from the saturated 0x3F80, so a regression that
#  drops into the saturation branch would be caught here. (At 3.984375
#  BF16 rounding itself lifts the answer to 0x3F80, so we can't use
#  that value to distinguish the two code paths.)
# ----------------------------------------------------------------

def test_tanh_interp_below_sat_boundary():
    x_bits = 0x4040
    x_f = _bf16_bits_to_f32(x_bits)
    got = tanh_bf16(x_bits, LUT)
    want = _f32_to_bf16_bits(math.tanh(x_f))
    assert abs(got - want) <= 2, (
        f"tanh({x_f}) got=0x{got:04X} want≈0x{want:04X}"
    )
    assert got < 0x3F80, (
        f"unexpected saturation to 1.0 at x={x_f}: got=0x{got:04X}"
    )


# ----------------------------------------------------------------
#  Mid-range cross-check against math.tanh. 32-entry LUT over [0, 4)
#  gives ~1% relative error once BF16 quantization is folded in; we
#  use 2% to leave room for ULP noise.
# ----------------------------------------------------------------

@pytest.mark.parametrize("x_f", [
    0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5,
    -0.125, -0.25, -0.5, -0.75, -1.0, -1.5, -2.0, -2.5, -3.0, -3.5,
])
def test_tanh_midrange_within_bf16_tolerance(x_f):
    bits = _f32_to_bf16_bits(x_f)
    got_f = _bf16_bits_to_f32(tanh_bf16(bits, LUT))
    want_f = math.tanh(_bf16_bits_to_f32(bits))
    rel = abs(got_f - want_f) / max(abs(want_f), 1e-3)
    assert rel < 2e-2, (
        f"tanh({x_f}): got={got_f} want={want_f} rel_err={rel:.3g}"
    )


# ----------------------------------------------------------------
#  compute_now: lane-for-lane wrap of `tanh_bf16` across 16 lanes.
# ----------------------------------------------------------------

def test_compute_now_matches_tanh_bf16_per_lane():
    box = TanhRec(P)
    inputs = [
        0x0000, 0x8000, 0x3F80, 0xBF80,  # ±0, ±1.0
        0x4080, 0xC080, 0x7F80, 0xFF80,  # ±4.0, ±inf
        0x3F00, 0xBF00, 0x4000, 0xC000,  # ±0.5, ±2.0
        0x407F, 0xC07F, 0x3E80, 0xBE80,  # ±3.984, ±0.25
    ]
    assert len(inputs) == N
    resp = box.compute_now(TanhReq(xVec=inputs))
    for i, bits in enumerate(inputs):
        want = tanh_bf16(bits, LUT)
        assert resp.result[i] == want, (
            f"lane {i}: in=0x{bits:04X} got=0x{resp.result[i]:04X} "
            f"want=0x{want:04X}"
        )


def test_compute_now_lane_mask_zeroes_disabled_lanes():
    box = TanhRec(P)
    inputs = [0x3F80] * N  # every lane BF16 1.0
    # Disable lanes 0, 3, 5, 15.
    mask = 0xFFFF & ~((1 << 0) | (1 << 3) | (1 << 5) | (1 << 15))
    resp = box.compute_now(TanhReq(xVec=inputs, laneMask=mask))
    want_active = tanh_bf16(0x3F80, LUT)
    for i in range(N):
        if (mask >> i) & 1:
            assert resp.result[i] == want_active, f"lane {i} enabled but wrong"
        else:
            assert resp.result[i] == 0x0000, (
                f"lane {i} disabled but got 0x{resp.result[i]:04X}"
            )


def test_compute_now_rejects_wrong_lane_count():
    box = TanhRec(P)
    with pytest.raises(ValueError, match="must have"):
        box.compute_now(TanhReq(xVec=[0] * (N - 1)))


# ----------------------------------------------------------------
#  step() + reset(): latency = 1 register stage.
# ----------------------------------------------------------------

def test_latencies_expose_tanh_with_latency_one():
    assert TanhRec.LATENCIES == {"tanh": 1}


def test_step_latency_is_one_and_drains_correctly():
    """First step(req) returns None (queue starts empty). Second
    step(req2) returns req1's result. A trailing drain step(None)
    pops req2."""
    box = TanhRec(P)
    req1 = TanhReq(xVec=[0x3F80] * N)  # tanh(1.0)
    req2 = TanhReq(xVec=[0x4000] * N)  # tanh(2.0)

    assert box.step("tanh", req1) is None

    second = box.step("tanh", req2)
    assert second is not None
    assert second.result == [tanh_bf16(0x3F80, LUT)] * N

    third = box.step("tanh", None)
    assert third is not None
    assert third.result == [tanh_bf16(0x4000, LUT)] * N


def test_reset_clears_queue():
    box = TanhRec(P)
    # Prime the queue with a real request, then wipe it.
    box.step("tanh", TanhReq(xVec=[0x3F80] * N))
    box.reset()
    # After reset, a drain step must return None — the primed result
    # is gone, not held over.
    assert box.step("tanh", None) is None


def test_step_rejects_unknown_op_name():
    box = TanhRec(P)
    with pytest.raises(KeyError, match="tanh"):
        box.step("exp", TanhReq(xVec=[0] * N))
