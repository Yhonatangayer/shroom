#!/usr/bin/env python3
"""asm_bsm_aria_errors.py
====================
Analytical binaural error evaluation for BSM+MagLS vs ASM(N=1)+MagLS
using the ARIA microphone array ATFs.

Plots binaural magnitude MSE vs frequency for both methods (left & right ears).

Usage
-----
    python examples/asm_bsm_aria_errors.py
"""

import os
import sys
import numpy as np

HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))
sys.path.insert(0, REPO_ROOT)

from shroom.utils.sofa import load_sofa
from shroom.utils.file_utils import load_file
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.encoders.bsm import BSM
from shroom.encoders.asm import ASM
from shroom.acoustics.hrtf_processing import magls_hrtf
from asm_project.errors import asm_bin_magnitude_mse_error
from bsm_project.errors import bsm_mag_mse_error
from devutils.plot import loglog_plot

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARIA_PATH    = os.path.join(REPO_ROOT, "data", "sofa_arrays", "aria_atfs_fixed.sofa")
SH_ORDER     = 1       # ASM reference encoder order
MAGLS_CUTOFF = 1200.0  # Hz


def main():
    # ------------------------------------------------------------------
    # 1. Load ARIA array
    # ------------------------------------------------------------------
    print("Loading ARIA array ATFs...")
    aria = load_sofa(ARIA_PATH)
    fs    = aria.fs
    n_fft = aria.data.shape[-1]
    M, Q  = aria.data.shape[0], aria.data.shape[1]
    freqs     = np.fft.fftfreq(n_fft, 1.0 / fs)
    pos_freqs = np.fft.rfftfreq(n_fft, 1.0 / fs)
    duration  = n_fft / fs
    print(f"  fs={fs} Hz | n_fft={n_fft} | M={M} mics | Q={Q} source dirs")

    # ------------------------------------------------------------------
    # 2. Prepare HRTF on the ARIA source grid
    # ------------------------------------------------------------------
    print("Loading and preparing HRTF...")
    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(n_fft)
    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(aria.grid)
    space_hrtf = hrtf.copy()  # (2, Q, F) — freq + space domain

    # ------------------------------------------------------------------
    # 3. ARIA array in frequency domain
    # ------------------------------------------------------------------
    aria_freq = aria.copy()
    aria_freq.toFreq()  # (M, Q, F)

    # ------------------------------------------------------------------
    # 4. BSM + MagLS
    # ------------------------------------------------------------------
    print(f"Computing BSM + MagLS (cutoff={MAGLS_CUTOFF} Hz)...")
    bsm = BSM(
        array=aria_freq,
        hrtf=space_hrtf,
        use_magls=True,
        magls_cutoff_frequency=MAGLS_CUTOFF,
        fs=fs,
        duration=duration,
    )
    cl, cr = bsm.get_coefficients()  # (F, M)

    # ------------------------------------------------------------------
    # 5. ASM (N=1) + MagLS
    # ------------------------------------------------------------------
    print(f"Computing ASM (N={SH_ORDER}) encoder...")
    asm = ASM(sh_order=SH_ORDER, array=aria_freq, fs=fs, duration=duration)
    cnm = asm.cnm.data  # (M, nm, F)

    print("Computing MagLS HRTF (SH domain)...")
    magls_result = magls_hrtf(hrtf=space_hrtf, sh_order=SH_ORDER, cutoff_over_freq=MAGLS_CUTOFF)
    magls_freq = magls_result.copy()
    magls_freq.toFreq()  # (2, nm, F)

    # ------------------------------------------------------------------
    # 6. Compute errors
    # ------------------------------------------------------------------
    print("Computing binaural errors...")

    bsm_err_l, bsm_err_r = bsm_mag_mse_error(cl, cr, aria_freq, space_hrtf, freqs)

    asm_err, _ = asm_bin_magnitude_mse_error(
        magls_freq.data, cnm, aria_freq.data, space_hrtf.data,
        freqs, return_variance=True,
    )  # asm_err: (2, F_pos)

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    print("Plotting...")
    loglog_plot(
        freqs=pos_freqs[1:],
        title="Binaural Magnitude MSE — ARIA Array (BSM+MagLS vs ASM+MagLS)",
        errors={
            "BSM + MagLS (L)":               bsm_err_l[1:],
            "BSM + MagLS (R)":               bsm_err_r[1:],
            f"ASM N={SH_ORDER} + MagLS (L)": asm_err[0, 1:],
            f"ASM N={SH_ORDER} + MagLS (R)": asm_err[1, 1:],
        },
        figsize=(10, 5),
        show=True,
    )
    print("Done.")


if __name__ == "__main__":
    main()
