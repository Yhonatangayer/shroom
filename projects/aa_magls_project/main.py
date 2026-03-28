import os
import numpy as np

from shroom.geometry.sampling import sphereicalGrid
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.utils.file_utils import load_file
from shroom.acoustics.spherical_array import SphericalArray
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.asm import ASM
from shroom.acoustics.hrtf_processing import array_aware_magls_hrtf, magls_hrtf
from asm_project.errors import asm_bin_magnitude_mse_error
from devutils.plot import loglog_plot

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    # 1. Setup
    fs = 48000
    duration = 0.0026666666666666666
    freqs = np.fft.fftfreq(int(duration * fs), 1 / fs)

    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(int(duration * fs))

    source_grid = from_fibonacci_grid(240)

    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(source_grid)
    space_hrtf = hrtf.copy()
    hrtf.toSH(1)
    hnm = hrtf.copy()

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

    asm = ASM(sh_order=1, array=array, fs=fs, duration=duration)
    cnm = asm.cnm

    hrtf_magls_result = magls_hrtf(
        hrtf=space_hrtf,
        sh_order=1,
        cutoff_over_freq=1200,
    )
    hrtf_magls_freq = hrtf_magls_result.copy()
    hrtf_magls_freq.toFreq()

    aa_magls_hrtf_result = array_aware_magls_hrtf(
        hrtf=space_hrtf,
        asm=asm,
        array=array,
        sh_order=1,
        cutoff_over_freq=1200,
    )
    aa_magls_hrtf_freq = aa_magls_hrtf_result.copy()
    aa_magls_hrtf_freq.toFreq()

    error_bin, var_bin = asm_bin_magnitude_mse_error(
        hnm.data, cnm.data, array.data, space_hrtf.data, freqs, return_variance=True
    )
    error_bin_magls, var_bin_magls = asm_bin_magnitude_mse_error(
        hrtf_magls_freq.data, cnm.data, array.data, space_hrtf.data, freqs, return_variance=True
    )
    error_bin_aa_magls, var_bin_aa_magls = asm_bin_magnitude_mse_error(
        aa_magls_hrtf_freq.data, cnm.data, array.data, space_hrtf.data, freqs, return_variance=True
    )

    pos_freqs = np.fft.rfftfreq(int(duration * fs), 1 / fs)

    loglog_plot(
        freqs=pos_freqs,
        title='ASM | Binaural Magnitude MSE: Complex vs MagLS vs AA-MagLS',
        errors={
            'complex left': error_bin[0, :],
            'complex right': error_bin[1, :],
            'magls left': error_bin_magls[0, :],
            'magls right': error_bin_magls[1, :],
            'aa-magls left': error_bin_aa_magls[0, :],
            'aa-magls right': error_bin_aa_magls[1, :],
        },
        figsize=(8, 5),
        show=True,
    )


if __name__ == "__main__":
    main()
