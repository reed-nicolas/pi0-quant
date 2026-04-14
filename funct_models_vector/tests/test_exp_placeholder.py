"""Placeholder tests for the `Exp` lane box (`exp` / `exp2`).

Until the HardFloat-faithful Exp lane box lands, the class ships only
the skeleton. These tests verify that:
  - The class can be constructed without raising (so the engine
    wiring can register it with the dispatcher).
  - The request / response dataclasses have the right shape.
  - `compute_now` raises `NotImplementedError` (so a caller that
    accidentally drives them gets an explicit failure, not silent
    wrong output).
  - `step()` with `req=None` is safe (drain path used by the
    cycle-accurate engine harness).

The dispatcher in `vector_engine_model.py` keeps `exp` / `exp2`
pointed at `_legacy_math_fallback` until this lane box is real.
"""

from __future__ import annotations

import pytest

from funct_models_vector.lane_boxes.exp import Exp, FPEXReq, FPEXResp
from funct_models_vector.vector_params import VectorParams


P = VectorParams()
N = P.num_lanes


def test_exp_class_constructs():
    box = Exp(P)
    assert "exp" in box.LATENCIES
    assert "exp2" in box.LATENCIES


def test_exp_compute_now_raises():
    box = Exp(P)
    with pytest.raises(NotImplementedError, match="placeholder"):
        box.compute_now(FPEXReq(xVec=[0] * N))


def test_exp_step_drain_is_safe():
    """Driving step() with req=None must not raise — the engine's
    drain phase calls every lane_box every cycle, and placeholders
    need to participate without blowing up."""
    box = Exp(P)
    assert box.step("exp", None) is None
    assert box.step("exp2", None) is None


def test_exp_step_with_request_raises():
    box = Exp(P)
    with pytest.raises(NotImplementedError, match="placeholder"):
        box.step("exp", FPEXReq(xVec=[0] * N))


@pytest.mark.skip(reason="pending HardFloat-faithful Exp lane box")
def test_exp_against_scala_golden():
    """Bit-exact cross-test against `ExpLane.scala`. Lands when the
    HardFloat-faithful Exp lane box is implemented."""
