"""
Precision regression tests for ASM, BSM, MagLS-HRTF, and AA-MagLS-HRTF filters.

Each test recomputes the filter from scratch using the *identical* deterministic
setup used by tests/reference_data/generate_filter_references.py and compares
against the saved .npz values at atol=1e-10.

If a reference file is missing, run:
    PYTHONPATH=src python tests/reference_data/generate_filter_references.py

Decoupling note: every filter here (ASM, BSM, MagLS, AA-MagLS) is computed from a
*synthetic* full-rank complex steering matrix, NOT from a physical
``SphericalArray``. These are filter-*algorithm* regression tests, so keeping them
independent of the array-generation physics means a change to the array model
(e.g. removing the per-order ghost-causing order mask in
``shroom.acoustics.physics``) does not break them. The physical array model is
covered separately in tests/test_spherical_array.py. A synthetic matrix is also
full-rank at DC, whereas a physical steering matrix is rank-1 there and crashes
the per-bin tikhonov solver.
"""

import pathlib
import pytest
import numpy as np

from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf, array_aware_magls_hrtf
from shroom.utils.grid_utils import from_fibonacci_grid

# ---------------------------------------------------------------------------
# Constants — must stay in sync with generate_filter_references.py
# ---------------------------------------------------------------------------
SEED = 42
FS = 8000
DURATION = 0.01
N_FFT = int(FS * DURATION)   # 80
Q = 50
N_MICS = 5
R_SPHERE = 0.08
SH_ORDER_SM = 3
ASM_SH_ORDER = 1
MAGLS_CUTOFF = 500.0

REF_DIR = pathlib.Path(__file__).parent / "reference_data"


def _ref(name: str) -> pathlib.Path:
    path = REF_DIR / name
    if not path.exists():
        pytest.skip(
            f"Reference file '{name}' not found. "
            "Run:  PYTHONPATH=src python tests/reference_data/generate_filter_references.py"
        )
    return path


# ---------------------------------------------------------------------------
# Module-scoped fixtures — built once per test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_setup():
    """Synthetic full-rank steering matrix + HRTFs — shared by every filter.

    Mirrors ``build_synthetic_setup`` in generate_filter_references.py exactly.
    No physical ``SphericalArray`` is built, so these fixtures are independent of
    the array-generation physics (covered by test_spherical_array.py).

    Keys
    ----
    array : synthetic full-rank steering matrix (M, Q, F), freq domain — ASM/BSM/AA.
    hrtf_mock : complex random HRTF (2, Q, F), freq domain — BSM.
    hrtf : real random HRTF (2, Q, F) in freq domain — MagLS / AA-MagLS.
    asm : ASM encoder built on the synthetic steering matrix.
    """
    source_grid = from_fibonacci_grid(Q)
    rng = np.random.default_rng(SEED)

    V = (rng.standard_normal((N_MICS, Q, N_FFT))
         + 1j * rng.standard_normal((N_MICS, Q, N_FFT)))
    array_mock = SpatialSignal(
        data=V, fs=FS, is_time=False, is_space=True, grid=source_grid
    )

    H = (rng.standard_normal((2, Q, N_FFT))
         + 1j * rng.standard_normal((2, Q, N_FFT)))
    hrtf_mock = SpatialSignal(
        data=H, fs=FS, is_time=False, is_space=True, grid=source_grid
    )

    hrtf_time = rng.standard_normal((2, Q, N_FFT))
    hrtf = SpatialSignal(
        data=hrtf_time, fs=FS, is_time=True, is_space=True, grid=source_grid
    )
    hrtf.toFreq()

    asm = ASM(sh_order=ASM_SH_ORDER, array=array_mock, fs=FS, duration=DURATION)
    return {"array": array_mock, "hrtf_mock": hrtf_mock, "hrtf": hrtf, "asm": asm}


# ---------------------------------------------------------------------------
# ASM
# ---------------------------------------------------------------------------

def test_asm_filter_precision(synthetic_setup):
    """ASM cnm matches saved reference to 1e-10."""
    ref = np.load(_ref("asm_filters.npz"))
    np.testing.assert_allclose(
        synthetic_setup["asm"].cnm.data, ref["cnm"],
        atol=1e-10, err_msg="ASM cnm deviates from reference",
    )


# ---------------------------------------------------------------------------
# BSM (no MagLS)
# ---------------------------------------------------------------------------

def test_bsm_filter_precision(synthetic_setup):
    """BSM cl/cr (no MagLS) match saved reference to 1e-10."""
    ref = np.load(_ref("bsm_filters.npz"))
    bsm = BSM(array=synthetic_setup["array"], hrtf=synthetic_setup["hrtf_mock"],
              use_magls=False, fs=FS)
    cl, cr = bsm.get_coefficients()
    np.testing.assert_allclose(cl, ref["cl"], atol=1e-10, err_msg="BSM cl deviates")
    np.testing.assert_allclose(cr, ref["cr"], atol=1e-10, err_msg="BSM cr deviates")


# ---------------------------------------------------------------------------
# BSM with MagLS
# ---------------------------------------------------------------------------

def test_bsm_magls_filter_precision(synthetic_setup):
    """BSM cl/cr (with MagLS) match saved reference to 1e-10."""
    ref = np.load(_ref("bsm_magls_filters.npz"))
    bsm = BSM(
        array=synthetic_setup["array"], hrtf=synthetic_setup["hrtf_mock"],
        use_magls=True, magls_cutoff_frequency=MAGLS_CUTOFF, fs=FS,
    )
    cl, cr = bsm.get_coefficients()
    np.testing.assert_allclose(cl, ref["cl"], atol=1e-10, err_msg="BSM-MagLS cl deviates")
    np.testing.assert_allclose(cr, ref["cr"], atol=1e-10, err_msg="BSM-MagLS cr deviates")


# ---------------------------------------------------------------------------
# MagLS HRTF
# ---------------------------------------------------------------------------

def test_magls_hrtf_precision(synthetic_setup):
    """MagLS HRTF SH coefficients match saved reference to 1e-10."""
    ref = np.load(_ref("magls_hrtf_filters.npz"))
    result = magls_hrtf(
        hrtf=synthetic_setup["hrtf"],
        sh_order=ASM_SH_ORDER,
        cutoff_over_freq=MAGLS_CUTOFF,
    )
    result.toFreq()
    np.testing.assert_allclose(
        result.data, ref["hnm"],
        atol=1e-10, err_msg="MagLS HRTF deviates from reference",
    )


# ---------------------------------------------------------------------------
# AA-MagLS HRTF
# ---------------------------------------------------------------------------

def test_aa_magls_hrtf_precision(synthetic_setup):
    """AA-MagLS HRTF SH coefficients match saved reference to 1e-10."""
    ref = np.load(_ref("aa_magls_hrtf_filters.npz"))
    result = array_aware_magls_hrtf(
        hrtf=synthetic_setup["hrtf"],
        asm=synthetic_setup["asm"],
        array=synthetic_setup["array"],
        sh_order=ASM_SH_ORDER,
        cutoff_over_freq=MAGLS_CUTOFF,
    )
    result.toFreq()
    np.testing.assert_allclose(
        result.data, ref["hnm"],
        atol=1e-10, err_msg="AA-MagLS HRTF deviates from reference",
    )
