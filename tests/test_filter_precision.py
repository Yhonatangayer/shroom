"""
Precision regression tests for ASM, BSM, MagLS-HRTF, and AA-MagLS-HRTF filters.

Each test recomputes the filter from scratch using the *identical* deterministic
setup used by tests/reference_data/generate_filter_references.py and compares
against the saved .npz values at atol=1e-10.

If a reference file is missing, run:
    PYTHONPATH=src python tests/reference_data/generate_filter_references.py

BSM note: physical SphericalArray steering matrices are rank-1 at DC, which
crashes tikhonov. BSM reference data is therefore generated from a synthetic
full-rank complex steering matrix (see generate_filter_references.py).
"""

import pathlib
import pytest
import numpy as np

from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf, array_aware_magls_hrtf
from shroom.geometry.sampling import sphereicalGrid
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
def asm_magls_setup():
    """Real SphericalArray + synthetic HRTF — for ASM, MagLS, AA-MagLS."""
    mics_grid = sphereicalGrid(
        az=np.deg2rad(np.array([-90, -45, 0, 45, 90])),
        co=np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90])),
    )
    source_grid = from_fibonacci_grid(Q)

    array = SphericalArray(
        fs=FS,
        duration=DURATION,
        r_sphere=R_SPHERE,
        r_mics=np.full(N_MICS, R_SPHERE),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=SH_ORDER_SM,
        convert_to_time=False,
        source_type="point_source",
    )

    rng = np.random.default_rng(SEED)
    hrtf_time = rng.standard_normal((2, Q, N_FFT))
    hrtf = SpatialSignal(
        data=hrtf_time, fs=FS, is_time=True, is_space=True, grid=source_grid
    )
    hrtf.toFreq()

    asm = ASM(sh_order=ASM_SH_ORDER, array=array, fs=FS, duration=DURATION)
    return {"array": array, "hrtf": hrtf, "asm": asm}


@pytest.fixture(scope="module")
def bsm_setup():
    """Synthetic full-rank steering matrix — for BSM (avoids DC rank-1 issue)."""
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
    return {"array": array_mock, "hrtf": hrtf_mock}


# ---------------------------------------------------------------------------
# ASM
# ---------------------------------------------------------------------------

def test_asm_filter_precision(asm_magls_setup):
    """ASM cnm matches saved reference to 1e-10."""
    ref = np.load(_ref("asm_filters.npz"))
    np.testing.assert_allclose(
        asm_magls_setup["asm"].cnm.data, ref["cnm"],
        atol=1e-10, err_msg="ASM cnm deviates from reference",
    )


# ---------------------------------------------------------------------------
# BSM (no MagLS)
# ---------------------------------------------------------------------------

def test_bsm_filter_precision(bsm_setup):
    """BSM cl/cr (no MagLS) match saved reference to 1e-10."""
    ref = np.load(_ref("bsm_filters.npz"))
    bsm = BSM(array=bsm_setup["array"], hrtf=bsm_setup["hrtf"], use_magls=False, fs=FS)
    cl, cr = bsm.get_coefficients()
    np.testing.assert_allclose(cl, ref["cl"], atol=1e-10, err_msg="BSM cl deviates")
    np.testing.assert_allclose(cr, ref["cr"], atol=1e-10, err_msg="BSM cr deviates")


# ---------------------------------------------------------------------------
# BSM with MagLS
# ---------------------------------------------------------------------------

def test_bsm_magls_filter_precision(bsm_setup):
    """BSM cl/cr (with MagLS) match saved reference to 1e-10."""
    ref = np.load(_ref("bsm_magls_filters.npz"))
    bsm = BSM(
        array=bsm_setup["array"], hrtf=bsm_setup["hrtf"],
        use_magls=True, magls_cutoff_frequency=MAGLS_CUTOFF, fs=FS,
    )
    cl, cr = bsm.get_coefficients()
    np.testing.assert_allclose(cl, ref["cl"], atol=1e-10, err_msg="BSM-MagLS cl deviates")
    np.testing.assert_allclose(cr, ref["cr"], atol=1e-10, err_msg="BSM-MagLS cr deviates")


# ---------------------------------------------------------------------------
# MagLS HRTF
# ---------------------------------------------------------------------------

def test_magls_hrtf_precision(asm_magls_setup):
    """MagLS HRTF SH coefficients match saved reference to 1e-10."""
    ref = np.load(_ref("magls_hrtf_filters.npz"))
    result = magls_hrtf(
        hrtf=asm_magls_setup["hrtf"],
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

def test_aa_magls_hrtf_precision(asm_magls_setup):
    """AA-MagLS HRTF SH coefficients match saved reference to 1e-10."""
    ref = np.load(_ref("aa_magls_hrtf_filters.npz"))
    result = array_aware_magls_hrtf(
        hrtf=asm_magls_setup["hrtf"],
        asm=asm_magls_setup["asm"],
        array=asm_magls_setup["array"],
        sh_order=ASM_SH_ORDER,
        cutoff_over_freq=MAGLS_CUTOFF,
    )
    result.toFreq()
    np.testing.assert_allclose(
        result.data, ref["hnm"],
        atol=1e-10, err_msg="AA-MagLS HRTF deviates from reference",
    )
