"""Byte-identity gate for `funct_models_vector.gen_vectors`.

The 6 per-family reference files under
`src/test/resources/vpu_test_vectors/vpu_<family>_vectors.txt` are not
tracked in git — the directory is in `.gitignore` — so "reference" here
means "the file currently on disk that `scripts/gen_vpu_test_vectors.py`
produced with the shipped default invocation (seed 12345, num 30,
num_lanes 16)." Any intentional regen also passes through this test.

Asserts that:

1. `gen_vectors.main(...)` is fully reproducible — two runs with the same
   seed produce identical output. This is a regression guard against the
   old `scripts/gen_vpu_vectors.py` bug where numpy's global RNG was
   un-seeded, causing sin/cos/tanh cases to drift between runs.
2. For each of the 6 families (binary, unary_simple, unary_math,
   row_reduce, col_reduce, vli) the per-family reference invocation
   reproduces `vpu_<family>_vectors.txt` byte-for-byte.
3. The `binary` reference file begins with the two hardcoded static add
   cases from `_static_cases()` — a sanity check that the static-case
   path is still firing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from funct_models_vector.gen_vectors import main


REPO_ROOT = Path(__file__).resolve().parents[5]
REF_DIR = REPO_ROOT / "src" / "test" / "resources" / "vpu_test_vectors"

# These must match the defaults in `scripts/gen_vpu_test_vectors.py`.
REF_SEED = 12345
REF_NUM = 30
REF_NUM_LANES = 16

# Family → ops mapping. Must match
# `scripts/gen_vpu_test_vectors.py:FAMILIES` exactly — if you change one,
# change the other. fp8pack / fp8unpack are deliberately excluded; their
# 2→1 / 1→2 phasing does not fit the single-pulse vector file format.
FAMILIES: dict[str, list[str]] = {
    "binary":       ["add", "sub", "mul", "pairmax", "pairmin"],
    "unary_simple": ["mov", "relu", "rcp"],
    "unary_math":   ["sqrt", "log", "exp", "exp2", "square", "cube", "sin", "cos", "tanh"],
    "row_reduce":   ["rsum", "rmin", "rmax"],
    "col_reduce":   ["csum", "cmin", "cmax"],
    "vli":          ["vliOne", "vliRow", "vliCol", "vliAll"],
}


def _ref_path(family: str) -> Path:
    return REF_DIR / f"vpu_{family}_vectors.txt"


def _family_argv(family: str, out: Path) -> list[str]:
    return [
        "--out", str(out),
        "--seed", str(REF_SEED),
        "--num", str(REF_NUM),
        "--num-lanes", str(REF_NUM_LANES),
        "--ops", *FAMILIES[family],
    ]


def _run_main(tmp_path: Path, argv: list[str]) -> Path:
    out = tmp_path / "vectors.txt"
    rc = main(["--out", str(out), *argv])
    assert rc == 0, f"main() returned non-zero: {rc}"
    assert out.is_file(), f"main() did not write {out}"
    return out


def test_ref_files_exist() -> None:
    missing = [str(_ref_path(f)) for f in FAMILIES if not _ref_path(f).is_file()]
    assert not missing, (
        "Per-family reference files missing. Regenerate with:\n"
        "  python3 scripts/gen_vpu_test_vectors.py\n"
        "Missing:\n  " + "\n  ".join(missing)
    )


def test_main_is_reproducible(tmp_path: Path) -> None:
    """Two runs with the same seed MUST produce identical output.

    Regression guard: the old `scripts/gen_vpu_vectors.py` only seeded
    Python's stdlib `random`, not `numpy.random`, so sin/cos/tanh cases
    drifted between runs.
    """
    shared = ["--seed", str(REF_SEED), "--num", "20"]
    a = _run_main(tmp_path / "a", shared)
    b = _run_main(tmp_path / "b", shared)
    assert a.read_bytes() == b.read_bytes(), (
        "gen_vectors.main() is non-deterministic under a fixed seed. "
        "Check that np.random.seed(args.seed) is still being called."
    )


@pytest.mark.parametrize("family", list(FAMILIES.keys()))
def test_matches_per_family_reference(tmp_path: Path, family: str) -> None:
    """Each `vpu_<family>_vectors.txt` must be reproducible byte-for-byte
    from the seed=12345 num=30 num-lanes=16 invocation with the family's
    op list. A drift here means either (a) gen_vectors changed its
    output format / RNG consumption, (b) a lane box changed its LUT, or
    (c) the on-disk reference is stale — in which case the right fix is
    `python3 scripts/gen_vpu_test_vectors.py --families {family}`."""
    ref = _ref_path(family)
    if not ref.is_file():
        pytest.skip(f"{ref} not on disk; run gen_vpu_test_vectors.py first")
    out = tmp_path / f"vpu_{family}_vectors.txt"
    rc = main(_family_argv(family, out))
    assert rc == 0, f"main() returned non-zero: {rc}"
    got = out.read_bytes()
    want = ref.read_bytes()
    if got != want:
        pytest.fail(
            f"gen_vectors.main() output for family={family!r} drifted from "
            f"{ref.name}. If intentional, regen with:\n"
            f"  python3 scripts/gen_vpu_test_vectors.py --families {family}\n"
            f"len(got)={len(got)}, len(want)={len(want)}"
        )


def test_binary_reference_starts_with_static_add_cases() -> None:
    """Sanity: `_static_cases()` emits ids 0 and 1 as hardcoded add cases
    when 'add' is in the op list. The `binary` family includes add, so
    its reference file must start with those two cases."""
    ref = _ref_path("binary")
    if not ref.is_file():
        pytest.skip(f"{ref} not on disk; run gen_vpu_test_vectors.py first")
    lines = ref.read_text().splitlines()
    assert lines[0].startswith('# 0 - "1.0 + 0.0 = 1.0'), (
        f"{ref.name} first line was {lines[0]!r}, expected static case 0"
    )
    assert any(l.startswith('# 1 - "1.0 + 2.0 = 3.0') for l in lines[:20]), (
        f"{ref.name} missing static case id=1 in first 20 lines"
    )
