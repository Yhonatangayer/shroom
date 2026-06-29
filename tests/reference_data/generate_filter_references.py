"""
Generate reference .npz files for filter precision regression tests.

Run once from the repo root:
    PYTHONPATH=src python tests/reference_data/generate_filter_references.py

The saved values are then compared against fresh computations in
tests/test_filter_precision.py at atol=1e-10.

Decoupling note: ASM, BSM and AA-MagLS filters are all computed from a single
*synthetic* full-rank complex steering matrix, NOT from a physical
``SphericalArray``. This keeps the filter-algorithm regression tests independent
of the array-generation physics (radial functions / damping in
``shroom.acoustics.physics``), which is exercised separately in
``tests/test_spherical_array.py``. Two practical benefits:

* a physical steering matrix is rank-1 at DC (all Bn collapse to the n=0 term),
  which the per-bin tikhonov solver does not guard against; the synthetic matrix
  is full-rank at every bin.
* changing the array model (e.g. removing the per-order ghost-causing order mask)
  no longer forces these fixtures to be regenerated.
"""

import pathlib
import numpy as np

from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf, array_aware_magls_hrtf
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


def build_synthetic_setup():
    """Synthetic full-rank steering matrix + two HRTFs, all on one source grid.

    The steering matrix ``array_mock`` is shared by ASM, BSM and AA-MagLS so none
    of the filter fixtures depend on the physical array model. ``hrtf`` is a real
    time-domain random HRTF (for MagLS / AA-MagLS, which need ``toSH``/``toTime``);
    ``hrtf_mock`` is a frequency-domain complex random HRTF (for BSM).
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

    return array_mock, hrtf_mock, hrtf


def main():
    print("Building synthetic setup …")
    array_mock, hrtf_mock, hrtf = build_synthetic_setup()

    # 1. ASM filters — synthetic matrix
    print("Computing ASM filters …")
    asm = ASM(sh_order=ASM_SH_ORDER, array=array_mock, fs=FS, duration=DURATION)
    cnm = asm.cnm.data.copy()
    np.savez_compressed(OUT_DIR / "asm_filters.npz", cnm=cnm)
    print(f"  saved asm_filters.npz  cnm{cnm.shape}")

    # 2. BSM filters (no MagLS) — synthetic matrix
    print("Computing BSM filters …")
    bsm = BSM(array=array_mock, hrtf=hrtf_mock, use_magls=False, fs=FS)
    cl, cr = bsm.get_coefficients()
    np.savez_compressed(OUT_DIR / "bsm_filters.npz", cl=cl, cr=cr)
    print(f"  saved bsm_filters.npz  cl{cl.shape}  cr{cr.shape}")

    # 3. BSM filters with MagLS — synthetic matrix
    print("Computing BSM-MagLS filters …")
    bsm_magls = BSM(
        array=array_mock,
        hrtf=hrtf_mock,
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

    # 5. AA-MagLS HRTF — synthetic matrix
    print("Computing AA-MagLS HRTF …")
    hrtf_aa = array_aware_magls_hrtf(
        hrtf=hrtf, asm=asm, array=array_mock,
        sh_order=ASM_SH_ORDER, cutoff_over_freq=MAGLS_CUTOFF,
    )
    hrtf_aa.toFreq()
    hnm_aa = hrtf_aa.data.copy()
    np.savez_compressed(OUT_DIR / "aa_magls_hrtf_filters.npz", hnm=hnm_aa)
    print(f"  saved aa_magls_hrtf_filters.npz  hnm{hnm_aa.shape}")

    print("\nAll reference files written to", OUT_DIR)


if __name__ == "__main__":
    main()
