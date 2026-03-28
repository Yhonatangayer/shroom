import pytest
import numpy as np
from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
# from spaudiopy.grids import load_lebedev
from shroom.utils.dsp_utils import (
    is_signal_frequency_sh_valid,
    is_signal_frequency_symmetric,
    is_sh_valid,
)


@pytest.fixture
def basic_array():
    """Fixture for a basic spherical array."""
    fs = 48000
    radius = 0.1
    n_mics = 6

    # Use co=0 (North Pole) to match legacy/standard conventions if needed,
    # or pi/2 (Equator). Let's use Equator as it's more common for simple rings.
    mics_grid = sphereicalGrid(
        az=np.linspace(0, 2 * np.pi, n_mics, endpoint=False),
        co=np.full(n_mics, np.pi / 2),
    )

    source_grid = from_fibonacci_grid(974)

    return SphericalArray(
        fs=fs,
        duration=0.01,
        r_sphere=radius,
        r_mics=np.full(n_mics, radius),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=14,
        convert_to_time=False,
    )


def test_array_initialization(basic_array):
    """Test array initialization and properties."""
    assert basic_array.fs == 48000
    assert basic_array.n_channels == 6  # 6 mics
    assert basic_array.is_space  # It's a spatial signal (mic signals)
    assert basic_array.grid.n_points == 974  # Lebedev 53 -> 974 points


def test_array_data_shape(basic_array):
    """Test the shape of the generated array manifold/data."""
    n_mics = 6
    n_sources = 974  # Lebedev degree 53 -> 974 points

    assert basic_array.data.shape[0] == n_mics
    assert basic_array.data.shape[1] == n_sources
    assert basic_array.data.shape[2] > 0  # Time samples


def test_bn_not_zeroed_at_sub_unit_k():
    """Regression: Bn n>=1 must not be zeroed at k < 1 rad/m (non-DC bins).

    With fs=48000 and nFFT=1024, index 1 is f=46.875 Hz → k=0.858 rad/m < 1.
    The old code used `k < 1` as the zero-mask, incorrectly killing n>=1 modes
    at this bin and making the steering matrix rank-1 there.
    """
    from shroom.acoustics.physics import _compute_bn_diagonal

    fs, nFFT = 48000, 1024
    pos_freqs = np.fft.rfftfreq(nFFT, 1 / fs)
    k = 2 * np.pi * pos_freqs / 343.0

    assert k[1] < 1.0, "test assumption: k[1] must be < 1 rad/m for this setup"

    bn = _compute_bn_diagonal(
        N=5, k=k, a=0.1, r_m=0.1,
        sphere_type="rigid", source_type="plane_wave", apply_damping=False,
    )
    # Row 1 corresponds to the first m=0 component of n=1 (ACN index 1)
    assert np.abs(bn[1, 1]) > 1e-6, (
        f"Bn n=1 at k[1]={k[1]:.4f} rad/m should be non-zero, got {bn[1, 1]}"
    )


def test_array_signal_validity(basic_array):
    """Test signal validity (symmetry, space/SH properties)."""
    # Data is (Mics, Sources, Freq)
    assert basic_array.is_freq
    assert basic_array.is_space
    # FFT along time axis
    data_space_freq = basic_array.data

    # 1. Check Symmetry (Real time signal)
    # Check for a single channel/source combination
    assert is_signal_frequency_symmetric(data_space_freq, freq_axis=-1)

    # 2. Check Space Validity (should be same as symmetric for space domain)
    # assert is_signal_frequency_space_valid(data_space_freq, freq_axis=-1)

    # 3. Check SH Validity
    array_sh = basic_array.copy()
    array_sh.toSH(N_sp=1)

    # 3.1
    pY = basic_array.grid.pinvY(1)
    Y = basic_array.grid.Y(1)
    assert is_sh_valid(Y, sh_axis=1)
    assert is_sh_valid(pY, sh_axis=0)

    # FFT the SH data
    assert array_sh.is_sh
    assert array_sh.is_freq
    data_sh_freq = array_sh.data

    # Check SH validity for a specific source direction (e.g. source 0)
    assert is_signal_frequency_sh_valid(data_sh_freq, freq_axis=2, sh_axis=1)
