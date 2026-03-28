"""
Generate reference .npz files for filter precision regression tests.

Run once from the repo root:
    PYTHONPATH=src python tests/reference_data/generate_filter_references.py

The saved values are then compared against fresh computations in
tests/test_filter_precision.py at atol=1e-10.

BSM note: physical SphericalArray steering matrices are rank-1 at DC (f=0)
because all Bn radial functions collapse to the n=0 term. The tikhonov solver
in bsm.py does not guard against this, so BSM filters are computed from a
synthetic full-rank complex steering matrix instead.
"""

import pathlib
import numpy as np

from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf, array_aware_magls_hrtf
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid

OUT_DIR = pathlib.Path(__file__).parent

# ---------------------------------------------------------------------------
# Constants — must stay in sync with test_filter_precision.py
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


def build_array_and_hrtf():
    """Real SphericalArray + synthetic HRTF — used for ASM, MagLS, AA-MagLS."""
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
    )

    rng = np.random.default_rng(SEED)
    hrtf_time = rng.standard_normal((2, Q, N_FFT))
    hrtf = SpatialSignal(
        data=hrtf_time, fs=FS, is_time=True, is_space=True, grid=source_grid
    )
    hrtf.toFreq()

    return array, hrtf, source_grid


def build_bsm_setup(source_grid):
    """Synthetic full-rank steering matrix + HRTF — used for BSM only."""
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
    return array_mock, hrtf_mock


def main():
    print("Building setup …")
    array, hrtf, source_grid = build_array_and_hrtf()
    array_bsm, hrtf_bsm = build_bsm_setup(source_grid)

    # 1. ASM filters
    print("Computing ASM filters …")
    asm = ASM(sh_order=ASM_SH_ORDER, array=array, fs=FS, duration=DURATION)
    cnm = asm.cnm.data.copy()
    np.savez_compressed(OUT_DIR / "asm_filters.npz", cnm=cnm)
    print(f"  saved asm_filters.npz  cnm{cnm.shape}")

    # 2. BSM filters (no MagLS) — synthetic matrix
    print("Computing BSM filters …")
    bsm = BSM(array=array_bsm, hrtf=hrtf_bsm, use_magls=False, fs=FS)
    cl, cr = bsm.get_coefficients()
    np.savez_compressed(OUT_DIR / "bsm_filters.npz", cl=cl, cr=cr)
    print(f"  saved bsm_filters.npz  cl{cl.shape}  cr{cr.shape}")

    # 3. BSM filters with MagLS — synthetic matrix
    print("Computing BSM-MagLS filters …")
    bsm_magls = BSM(
        array=array_bsm,
        hrtf=hrtf_bsm,
        use_magls=True,
        magls_cutoff_frequency=MAGLS_CUTOFF,
        fs=FS,
    )
    cl_m, cr_m = bsm_magls.get_coefficients()
    np.savez_compressed(OUT_DIR / "bsm_magls_filters.npz", cl=cl_m, cr=cr_m)
    print(f"  saved bsm_magls_filters.npz  cl{cl_m.shape}  cr{cr_m.shape}")

    # 4. MagLS HRTF
    print("Computing MagLS HRTF …")
    hrtf_magls = magls_hrtf(hrtf=hrtf, sh_order=ASM_SH_ORDER, cutoff_over_freq=MAGLS_CUTOFF)
    hrtf_magls.toFreq()
    hnm_magls = hrtf_magls.data.copy()
    np.savez_compressed(OUT_DIR / "magls_hrtf_filters.npz", hnm=hnm_magls)
    print(f"  saved magls_hrtf_filters.npz  hnm{hnm_magls.shape}")

    # 5. AA-MagLS HRTF
    print("Computing AA-MagLS HRTF …")
    hrtf_aa = array_aware_magls_hrtf(
        hrtf=hrtf, asm=asm, array=array,
        sh_order=ASM_SH_ORDER, cutoff_over_freq=MAGLS_CUTOFF,
    )
    hrtf_aa.toFreq()
    hnm_aa = hrtf_aa.data.copy()
    np.savez_compressed(OUT_DIR / "aa_magls_hrtf_filters.npz", hnm=hnm_aa)
    print(f"  saved aa_magls_hrtf_filters.npz  hnm{hnm_aa.shape}")

    print("\nAll reference files written to", OUT_DIR)


if __name__ == "__main__":
    main()
