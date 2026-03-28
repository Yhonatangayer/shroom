import os
import numpy as np

from shroom.geometry.sampling import sphereicalGrid
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.utils.file_utils import load_file
from shroom.acoustics.spherical_array import SphericalArray
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.encoders.asm import ASM
from asm_project.errors import asm_mse_error, asm_bin_mse_error, linear_spectral_error
from devutils.plot import loglog_plot

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


def main():
    # 1. Setup
    fs = 16000
    duration = 0.016
    freqs = np.fft.fftfreq(int(duration*fs), 1/fs)

    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(int(duration*fs))

    source_grid = from_fibonacci_grid(240)

    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(source_grid)
    space_hrtf = hrtf.copy()
    hrtf.toSH(1)
    hnm = hrtf.copy()

    az = np.deg2rad(np.array([-90,-45,0,45,90]))
    co = np.deg2rad(np.array([90, 90+18, 90-18, 90+18, 90]))
    mic_grid = sphereicalGrid(az=az, co=co)

    array = SphericalArray(
        fs = fs,
        duration = duration,
        r_sphere =  0.08,
        r_mics =  0.08*np.ones((mic_grid.n_points,)),
        source_grid = source_grid,
        mics_grid = mic_grid,
        sphere_type = "rigid",
        sh_order_for_sm_calc = 14,
        convert_to_time = False,
    )

    asm = ASM(sh_order=1, array=array, fs=fs, duration=duration)
    cnm = asm.cnm

    error_mse, var_mse = asm_mse_error(cnm.data, array.data, array.grid.Y(1), freqs, return_variance=True)
    lse, var_lse = linear_spectral_error(cnm.data, array.data, array.grid.Y(1), freqs, return_variance=True)
    error_bin, var_bin = asm_bin_mse_error(hnm.data, cnm.data, array.data, space_hrtf.data, freqs, return_variance=True)

    pos_freqs = np.fft.rfftfreq(int(duration*fs), 1/fs)

    loglog_plot(
        freqs=pos_freqs,
        title='ASM | Complex MSE per SH Channel',
        errors={
            '(0,0)': error_mse[0, ...],
            '(1,-1)': error_mse[1, ...],
            '(1,0)': error_mse[2, ...],
            '(1,1)': error_mse[3, ...],
        },
        variances={
            '(0,0)': var_mse[0, ...],
            '(1,-1)': var_mse[1, ...],
            '(1,0)': var_mse[2, ...],
            '(1,1)': var_mse[3, ...],
        },
        figsize=(6, 4),
        ylim=(-30, 5),
        show=True,
    )

    loglog_plot(
        freqs=pos_freqs,
        title='ASM | Log Spectral Error per SH Channel',
        errors={
            '(0,0)': lse[0, ...],
            '(1,-1)': lse[1, ...],
            '(1,0)': lse[2, ...],
            '(1,1)': lse[3, ...],
        },
        variances={
            '(0,0)': var_lse[0, ...],
            '(1,-1)': var_lse[1, ...],
            '(1,0)': var_lse[2, ...],
            '(1,1)': var_lse[3, ...],
        },
        figsize=(6, 4),
        ylim=(-30, 5),
        show=True,
    )

    loglog_plot(
        freqs=pos_freqs,
        title='ASM | Binaural Complex MSE',
        errors={
            'left': error_bin[0, :],
            'right': error_bin[1, :],
        },
        variances={
            'left': var_bin[0, :],
            'right': var_bin[1, :],
        },
        figsize=(6, 4),
        show=True,
    )

if __name__ == "__main__":
    main()
