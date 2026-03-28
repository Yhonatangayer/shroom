import os
import numpy as np
import scipy.io
from shroom.geometry.sampling import sphereicalGrid
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.utils.file_utils import load_file
from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.bsm import BSM
from bsm_project.errors import bsm_mse_error, bsm_mag_mse_error
from devutils.plot import loglog_plot

MAT_PATH = '/Users/yhonag/Downloads/semiCirc_M6_ATF.mat'

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def main():
    # --- Setup (parameters matched to MATLAB .mat file) ---
    fs = 48000
    mat = scipy.io.loadmat(MAT_PATH)
    mat_data = mat['out_data'][0, 0]
    nFFT = int(mat_data['nFFT'][0, 0])
    duration = nFFT / fs
    Omega = mat_data['Omega']               # (1730, 3): [co, az, r=1]

    freqs = np.fft.fftfreq(nFFT, 1 / fs)
    pos_freqs = np.fft.rfftfreq(nFFT, 1 / fs)

    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(fs)
    hrtf.zero_pad(nFFT)

    source_grid = sphereicalGrid(az=Omega[:, 1], co=Omega[:, 0])

    hrtf.toFreq()
    hrtf.toSH(40)
    hrtf.toSpace(source_grid)
    space_hrtf = hrtf.copy()

    az = np.deg2rad(np.array([295.7143, 321.4286, 347.1429, 12.8571, 38.5714, 64.2857]))
    co = np.deg2rad(np.array([90, 90, 90, 90, 90, 90]))
    mic_grid = sphereicalGrid(az=az, co=co)

    array = SphericalArray(
        fs=fs,
        duration=duration,
        r_sphere=0.1,
        r_mics=0.1 * np.ones((mic_grid.n_points,)),
        source_grid=source_grid,
        mics_grid=mic_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=40,
        convert_to_time=False,
        source_type="plane_wave",
        apply_damping=False,
        normalize_columns=True,
    )

    # BSM without MagLS
    bsm_reg = BSM(
        array=array,
        hrtf=space_hrtf,
        use_magls=False,
        fs=fs,
        duration=duration,
    )
    cl_reg, cr_reg = bsm_reg.get_coefficients()

    # BSM with MagLS
    bsm_mag = BSM(
        array=array,
        hrtf=space_hrtf,
        use_magls=True,
        magls_cutoff_frequency=800.0,
        fs=fs,
        duration=duration,
    )
    cl_mag, cr_mag = bsm_mag.get_coefficients()

    # Compute errors + variance
    mse_l_reg, mse_r_reg, var_l_reg, var_r_reg = bsm_mse_error(
        cl_reg, cr_reg, array, space_hrtf, freqs, return_variance=True
    )
    mse_l_mag, mse_r_mag, var_l_mag, var_r_mag = bsm_mag_mse_error(
        cl_mag, cr_mag, array, space_hrtf, freqs, return_variance=True
    )

    loglog_plot(
        freqs=pos_freqs,
        title='BSM | Complex MSE',
        errors={
            'left': mse_l_reg,
            'right': mse_r_reg,
        },
        variances={
            'left': var_l_reg,
            'right': var_r_reg,
        },
        figsize=(8, 5),
        show=True,
    )

    loglog_plot(
        freqs=pos_freqs,
        title='BSM MagLS | Magnitude MSE',
        errors={
            'left': mse_l_mag,
            'right': mse_r_mag,
        },
        variances={
            'left': var_l_mag,
            'right': var_r_mag,
        },
        figsize=(8, 5),
        show=True,
    )


if __name__ == "__main__":
    main()
