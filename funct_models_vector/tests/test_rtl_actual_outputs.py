"""Per-family RTL-snapshot cross-test: funct model vs. captured runs of
the Scala `VectorEngineTop<Family>VectorTest` drivers.

For each family `F` in {binary, unary_simple, unary_math, row_reduce,
col_reduce, vli}, parses

    src/test/resources/vpu_test_vectors/vpu_<F>_vectors.txt          (golden inputs)
    src/test/resources/vpu_test_vectors/vpu_<F>_rtl_outputs.txt      (RTL snapshot)

and compares, per (case_id, lane), the RTL's `actual_hex` against
`VectorEngineModel.execute(...)` on the same input row. This is the only
layer of the test suite that compares the funct model against actual
hardware output — every other layer (per-lane-box unit tests and torch
adapter tests) checks the funct model against a Python or torch
reference. A disagreement here means either (a) a funct model bug
against the RTL, or (b) an RTL bug relative to what the Python golden
says it should produce.

Regeneration: each RTL snapshot file is written out automatically by
the corresponding `VectorEngineTop<Family>VectorTest.scala` run, which
is the one and only source of truth for those files. If this test fails
with a case-id mismatch or a stale-op complaint, the usual fix is to
rerun the family test in mill.
"""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pytest

from funct_models_vector.vector_engine_model import VectorEngineModel
from funct_models_vector.vector_params import VectorParams
from funct_models_vector.vpu_vector_file import VPUTestCase, read_cases


REPO_ROOT = Path(__file__).resolve().parents[5]
VECTOR_DIR = REPO_ROOT / "src" / "test" / "resources" / "vpu_test_vectors"

# Keep this list in sync with `scripts/gen_vpu_test_vectors.py:FAMILIES`
# and with the six `VectorEngineTop<Family>VectorTest.scala` classes.
_FAMILIES: tuple[str, ...] = (
    "binary",
    "unary_simple",
    "unary_math",
    "row_reduce",
    "col_reduce",
    "vli",
)

# fp8pack / fp8unpack use a different golden-file format and are not
# part of any family bucket. Kept for documentation symmetry with the
# old monolith test.
_SKIP_OPS = frozenset({"fp8pack", "fp8unpack"})

_COL_OPS = frozenset({"csum", "cmax", "cmin"})
_VLI_OPS = frozenset({"vliOne", "vliCol", "vliRow", "vliAll"})
_BIN_OPS = frozenset({"add", "sub", "mul", "pairmax", "pairmin"})
_ROW_REDUCE_OPS = frozenset({"rsum", "rmax", "rmin"})


def _h(s: str) -> int:
    return int(s, 16) & 0xFFFF


# ----------------------------------------------------------------
#  rtl_outputs.txt parser (per-family file; same row format as the
#  old monolith rtl_actual_outputs.txt, minus the optional inA/inB
#  failure-context suffix which the new unified driver never emits).
# ----------------------------------------------------------------

@dataclass(frozen=True)
class RTLLaneRow:
    lane: int
    op: str
    actual_hex: int


def _parse_rtl_rows(path: Path) -> dict[int, dict[int, RTLLaneRow]]:
    """Returns `{case_id: {lane_idx: RTLLaneRow}}` deduped from the file.

    Raises if the same `(case_id, lane)` has two distinct snapshot rows —
    that would mean the file itself is internally inconsistent.
    """
    assert path.is_file(), f"Missing RTL snapshot {path}"
    lines = path.read_text().splitlines()
    rows: dict[int, dict[int, RTLLaneRow]] = defaultdict(dict)
    for raw in lines[1:]:  # skip header
        line = raw.rstrip()
        if not line.strip():
            continue
        toks = line.split()
        try:
            hex_idx = next(i for i, t in enumerate(toks) if t.startswith("0x"))
        except StopIteration:
            continue
        lane = int(toks[hex_idx - 2])
        op = toks[hex_idx - 1]
        actual_hex = int(toks[hex_idx], 16) & 0xFFFF
        case_id = int(toks[0])
        row = RTLLaneRow(lane=lane, op=op, actual_hex=actual_hex)
        prior = rows[case_id].get(lane)
        if prior is not None and prior != row:
            raise AssertionError(
                f"case {case_id} lane {lane}: inconsistent rtl rows "
                f"{prior!r} vs {row!r}"
            )
        rows[case_id][lane] = row
    return dict(rows)


# ----------------------------------------------------------------
#  Dispatch helper: route a parsed VPUTestCase through VectorEngineModel
# ----------------------------------------------------------------

def _dispatch(model: VectorEngineModel, case: VPUTestCase) -> list[int]:
    op = case.vpu_op
    num_lanes = case.num_lanes

    if op in _VLI_OPS:
        imm = _h(case.vec_a[0])
        result = model.execute(op, imm=imm)
        imm_hex = result[0] & 0xFFFF
        if op == "vliOne":
            return [imm_hex]
        if op == "vliRow":
            return [imm_hex] * num_lanes
        return [imm_hex] * (2 * num_lanes)

    a_bits = [_h(h) for h in case.vec_a]
    b_bits = [_h(h) for h in case.vec_b] if case.vec_b else []

    if op in _COL_OPS:
        return model.execute(op, a_vec=a_bits)
    if op in _ROW_REDUCE_OPS or op == "mov":
        return model.execute(op, a_vec=a_bits)
    if op in _BIN_OPS:
        return model.execute(op, a_vec=a_bits, b_vec=b_bits)
    return model.execute(op, a_vec=a_bits)


# Per-op tolerance buckets, mirroring
# `VpuVectorTestSupport.checkTolerance` so this drift guard matches the
# Scala family tests. Any (abs<=maxAbs) OR (rel<=maxRel) passes.
_TOLERANCE: dict[str, tuple[float, float]] = {
    "sin": (0.05, 0.05),
    "cos": (0.05, 0.05),
    "tanh": (0.05, 0.05),
    "log": (0.05, 0.05),
    "exp": (0.05, 0.05),
    "exp2": (0.05, 0.05),
    "sqrt": (0.05, 0.05),
    "cube": (0.02, 0.02),
    "square": (0.02, 0.02),
}
_DEFAULT_TOL = (0.01, 0.01)


def _bf16_bits_to_f32(bits: int) -> float:
    fp32 = (bits & 0xFFFF) << 16
    return struct.unpack("<f", struct.pack("<I", fp32))[0]


def _within_tolerance(op: str, got_bits: int, want_bits: int) -> bool:
    """Return True iff `got_bits` is within the Scala per-op tolerance
    bucket for `op` against `want_bits`. Inf/NaN equality is checked
    up front, mirroring VpuVectorTestSupport.checkTolerance."""
    if got_bits == want_bits:
        return True
    a = _bf16_bits_to_f32(got_bits)
    e = _bf16_bits_to_f32(want_bits)
    if math.isnan(a) and math.isnan(e):
        return True
    if math.isinf(a) and math.isinf(e) and (a > 0) == (e > 0):
        return True
    abs_err = abs(a - e)
    rel_err = 0.0 if e == 0.0 and a == 0.0 else (
        float("inf") if e == 0.0 else abs_err / abs(e)
    )
    max_rel, max_abs = _TOLERANCE.get(op, _DEFAULT_TOL)
    return abs_err <= max_abs or rel_err <= max_rel


def _gather_rtl_mismatches(
    model: VectorEngineModel,
    case: VPUTestCase,
    rtl_lanes: dict[int, RTLLaneRow],
) -> list[str]:
    got = _dispatch(model, case)
    out: list[str] = []
    for lane_idx, row in sorted(rtl_lanes.items()):
        if row.op != case.vpu_op:
            out.append(
                f"lane {lane_idx}: snapshot op={row.op!r}, current op={case.vpu_op!r}"
            )
            continue
        if lane_idx >= len(got):
            continue
        g = got[lane_idx] & 0xFFFF
        if not _within_tolerance(case.vpu_op, g, row.actual_hex):
            out.append(
                f"lane {lane_idx}: got 0x{g:04X}, want 0x{row.actual_hex:04X}"
            )
    return out


# ----------------------------------------------------------------
#  Per-family file loading + pytest parametrization
# ----------------------------------------------------------------

def _family_vectors_path(family: str) -> Path:
    return VECTOR_DIR / f"vpu_{family}_vectors.txt"


def _family_rtl_path(family: str) -> Path:
    return VECTOR_DIR / f"vpu_{family}_rtl_outputs.txt"


def _load_family_cases(family: str) -> list[VPUTestCase]:
    p = _family_vectors_path(family)
    if not p.is_file():
        return []
    return [c for c in read_cases(str(p)) if c.vpu_op not in _SKIP_OPS]


def _parametrize_cases() -> list[tuple[str, VPUTestCase]]:
    """One row per (family, case) that exists in both the golden and
    the RTL snapshot for that family."""
    out: list[tuple[str, VPUTestCase]] = []
    for fam in _FAMILIES:
        rtl_path = _family_rtl_path(fam)
        if not rtl_path.is_file():
            continue
        rtl_rows = _parse_rtl_rows(rtl_path)
        for case in _load_family_cases(fam):
            if case.case_id in rtl_rows:
                out.append((fam, case))
    return out


_STRICT_CASES = _parametrize_cases()


def _pp_id(fam: str, case: VPUTestCase) -> str:
    return f"{fam}-{case.case_id}-{case.vpu_op}"


@pytest.fixture(scope="module")
def model() -> VectorEngineModel:
    return VectorEngineModel(VectorParams())


def test_family_rtl_snapshots_present() -> None:
    """Each family should have both a golden and an RTL snapshot on
    disk. Run `python3 scripts/gen_vpu_test_vectors.py` and the six
    `mill atlas.test.testOnly atlas.vector.VectorEngineTop<Family>VectorTest`
    targets to regenerate whichever is missing."""
    missing: list[str] = []
    for fam in _FAMILIES:
        if not _family_vectors_path(fam).is_file():
            missing.append(str(_family_vectors_path(fam)))
        if not _family_rtl_path(fam).is_file():
            missing.append(str(_family_rtl_path(fam)))
    assert not missing, "Missing per-family files:\n  " + "\n  ".join(missing)


def test_family_case_alignment() -> None:
    """For each family, the golden cases and RTL snapshot cases should
    have the same case-id set. A divergence means someone regenerated
    one of the two without the other."""
    errors: list[str] = []
    for fam in _FAMILIES:
        rtl_path = _family_rtl_path(fam)
        if not rtl_path.is_file():
            continue
        rtl_rows = _parse_rtl_rows(rtl_path)
        golden_ids = {c.case_id for c in _load_family_cases(fam)}
        rtl_ids = set(rtl_rows.keys())
        missing_in_rtl = golden_ids - rtl_ids
        missing_in_golden = rtl_ids - golden_ids
        if missing_in_rtl:
            errors.append(
                f"[{fam}] golden has case_ids not in rtl snapshot: "
                f"{sorted(missing_in_rtl)}"
            )
        if missing_in_golden:
            errors.append(
                f"[{fam}] rtl snapshot has case_ids not in golden: "
                f"{sorted(missing_in_golden)}"
            )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize(
    "family,case",
    _STRICT_CASES,
    ids=[_pp_id(fam, c) for fam, c in _STRICT_CASES],
)
def test_rtl_diff_strict(
    model: VectorEngineModel,
    family: str,
    case: VPUTestCase,
) -> None:
    rtl_rows = _parse_rtl_rows(_family_rtl_path(family))
    rtl_lanes = rtl_rows[case.case_id]
    mismatches = _gather_rtl_mismatches(model, case, rtl_lanes)
    if mismatches:
        head = "\n  ".join(mismatches[:8])
        more = f"\n  ... {len(mismatches) - 8} more" if len(mismatches) > 8 else ""
        pytest.fail(
            f"[{family}] case {case.case_id} ({case.vpu_op}) funct model != RTL "
            f"{_family_rtl_path(family).name}:\n  {head}{more}"
        )
