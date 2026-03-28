import pytest
import numpy as np
from shroom.encoders.asm import ASM, calculate_asm_coefficients
from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.utils.dsp_utils import is_signal_frequency_sh_valid


@pytest.fixture
def real_array_signal():
    """Create a real SphericalArray signal in frequency domain."""
    fs = 16000
    radius = 0.1
    n_mics = 6

    # Mics on equator
    mics_grid = sphereicalGrid(
        az=np.linspace(0, 2 * np.pi, n_mics, endpoint=False),
        co=np.full(n_mics, np.pi / 2),
    )

    # Source grid (Lebedev)
    source_grid = from_fibonacci_grid(50)  # 50 points

    array = SphericalArray(
        fs=fs,
        duration=0.01,
        r_sphere=radius,
        r_mics=np.full(n_mics, radius),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=3,
        convert_to_time=True,
    )

    # validation of representation
    array.toFreq()
    array_copy = array.copy()
    array_copy.toSH(N_sp=3)
    assert array_copy.is_sh and array_copy.is_freq
    assert is_signal_frequency_sh_valid(array_copy.data, freq_axis=-1)

    return array


def test_calculate_asm_coefficients(real_array_signal):
    """Test the standalone calculation function."""
    sm = real_array_signal.data
    sh_order = 1
    Y = real_array_signal.grid.Y(N_sp=sh_order)

    # sm: (M, Q, F)
    # Y: (Q, L)

    coeffs = calculate_asm_coefficients(sm, Y)

    # Expected shape: (L, M, F)
    L = (sh_order + 1) ** 2
    M = sm.shape[0]
    F = sm.shape[2]

    assert coeffs.shape == (M, L, F)
    assert coeffs.dtype == np.complex128


def test_asm_class(real_array_signal):
    """Test the ASM class wrapper."""
    sh_order = 1
    asm = ASM(
        sh_order=sh_order,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
    )

    # Test cnm representation
    cnm_signal = asm.calculate()

    assert isinstance(cnm_signal, SpatialSignal)  # Should be SpatialSignal
    assert cnm_signal.is_freq
    assert cnm_signal.is_sh  # SH domain

    L = (sh_order + 1) ** 2
    M = real_array_signal.n_channels
    F = real_array_signal.data.shape[2]

    assert cnm_signal.data.shape == (M, L, F)

def test_encode_amb(real_array_signal):
    """Test encoding microphone signals to Ambisonics."""
    sh_order = 1
    asm = ASM(
        sh_order=sh_order,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
    )

    # Force calculation
    asm.calculate()

    # Create mock mic signals (Time domain)
    # Shape: (Time, Mics)
    n_samples = 1000
    n_mics = real_array_signal.n_channels
    mic_signals = np.random.randn(n_samples, n_mics)

    encoded = asm.encode_amb(mic_signals)

    # encoded is SpatialSignal
    assert encoded.is_time
    assert encoded.is_sh

    L = (sh_order + 1) ** 2
    # Output data shape: (L, 1, Time) or (L, Time) depending on implementation
    # SpatialSignal data is (Channels, Grid, Time)
    # Here Channels=L, Grid=1?

    # Check shape
    assert encoded.data.shape[:2] == (1, L)
    encodedf = np.fft.fft(encoded.data, axis=-1)
    assert is_signal_frequency_sh_valid(encodedf, freq_axis=-1)
