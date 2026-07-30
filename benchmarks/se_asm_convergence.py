"""Spectrally-equalized ASM (SE-ASM) vs. plain ASM.

Validates the ``spectrally_equalized`` flag of :class:`shroom.encoders.asm.ASM`:
SE-ASM rescales every SH channel so that its linear spectral magnitude stays at
0 dB across the whole band, where plain ASM collapses towards zero above the
spatial-aliasing frequency of the array. The price is a larger complex MSE.

Three figures are produced (per-channel complex MSE, per-channel LSE, and
binaural magnitude MSE with a MagLS HRTF), all written to ``benchmarks/figures/``.
"""
import os

import numpy as np

from shroom.geometry.sampling import sphereicalGrid
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.utils.file_utils import load_file
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.asm import ASM
from shroom_dev.errors import asm_bin_magnitude_mse_error, asm_mse_error, linear_spectral_error
from shroom_dev.plot import loglog_plot

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

SH_LABELS = ["(0,0)", "(1,-1)", "(1,0)", "(1,1)"]
SH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
ENCODER_STYLES = {"ASM": "-", "SE-ASM": "--"}

SHOW = True


def _per_channel_curves(errors_by_encoder):
    """Flatten {encoder: (nm, F)} into loglog_plot dicts keyed by label."""
    errors, styles, colors = {}, {}, {}
    for encoder, err in errors_by_encoder.items():
        for i, nm in enumerate(SH_LABELS):
            label = f"{encoder} {nm}"
            errors[label] = err[i, ...]
            styles[label] = ENCODER_STYLES[encoder]
            colors[label] = SH_COLORS[i]
    return errors, styles, colors


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Setup
    fs = 48000
    duration = 512 / 48000
    n_fft = int(duration * fs)
    freqs = np.fft.fftfreq(n_fft, 1 / fs)
    pos_freqs = np.fft.rfftfreq(n_fft, 1 / fs)

    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(n_fft)

    source_grid = from_fibonacci_grid(240)

    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(source_grid)
    space_hrtf = hrtf.copy()

    az = np.deg2rad(np.array([-90, -45, 0, 45, 90]))
    co = np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90]))
    mic_grid = sphereicalGrid(az=az, co=co)

    array = SphericalArray(
        fs=fs,
        duration=duration,
        r_sphere=0.08,
        r_mics=0.08 * np.ones((mic_grid.n_points,)),
        source_grid=source_grid,
        mics_grid=mic_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=14,
        convert_to_time=False,
    )

    Y = array.grid.Y(1)

    # 2. Encoders: plain ASM and its spectrally-equalized counterpart
    asm = ASM(sh_order=1, array=array, fs=fs, duration=duration)
    se_asm = ASM(
        sh_order=1,
        array=array,
        fs=fs,
        duration=duration,
        spectrally_equalized=True,
    )
    cnm = {"ASM": asm.cnm.data, "SE-ASM": se_asm.cnm.data}

    # 3. Errors
    mse = {name: asm_mse_error(c, array.data, Y, freqs) for name, c in cnm.items()}
    lse = {name: linear_spectral_error(c, array.data, Y, freqs) for name, c in cnm.items()}

    hrtf_magls = magls_hrtf(hrtf=space_hrtf.copy(), sh_order=1, cutoff_over_freq=1200)
    hrtf_magls.toFreq()
    bin_mse = {
        name: asm_bin_magnitude_mse_error(
            hrtf_magls.data, c, array.data, space_hrtf.data, freqs
        )
        for name, c in cnm.items()
    }

    # 4. Plots
    errors, styles, colors = _per_channel_curves(mse)
    loglog_plot(
        freqs=pos_freqs,
        title="ASM vs SE-ASM | Complex MSE per SH Channel",
        errors=errors,
        styles=styles,
        colors=colors,
        figsize=(7, 4),
        ylim=(-30, 20),
        save_path=os.path.join(FIGURES_DIR, "se_asm_mse.png"),
        show=SHOW,
    )

    errors, styles, colors = _per_channel_curves(lse)
    loglog_plot(
        freqs=pos_freqs,
        title="ASM vs SE-ASM | Linear Spectral Error per SH Channel",
        errors=errors,
        styles=styles,
        colors=colors,
        figsize=(7, 4),
        ylim=(-30, 20),
        save_path=os.path.join(FIGURES_DIR, "se_asm_lse.png"),
        show=SHOW,
    )

    loglog_plot(
        freqs=pos_freqs,
        title="ASM vs SE-ASM | Binaural Magnitude MSE (MagLS HRTF)",
        errors={
            f"{name} {ear}": err[i, :]
            for name, err in bin_mse.items()
            for i, ear in enumerate(["left", "right"])
        },
        styles={
            f"{name} {ear}": ENCODER_STYLES[name]
            for name in bin_mse
            for ear in ["left", "right"]
        },
        figsize=(7, 4),
        save_path=os.path.join(FIGURES_DIR, "se_asm_binaural_magnitude_mse.png"),
        show=SHOW,
    )


if __name__ == "__main__":
    main()
