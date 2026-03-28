import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.geometry.sampling import sphereicalGrid


@pytest.fixture
def mock_grid():
    """A simple grid with 4 points on the equator."""
    az = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    co = np.full(4, np.pi / 2)
    return sphereicalGrid(az, co)


@pytest.fixture
def mock_signal(mock_grid):
    """A simple spatial signal."""
    fs = 48000
    n_samples = 1000
    n_channels = 1  # Mono per grid point

    # Create data: (Channels, Grid, Time)
    data = np.zeros((n_channels, mock_grid.n_points, n_samples))
    data[0, :, 0] = 1.0  # Impulse at t=0 for all points

    return SpatialSignal(data, fs, is_time=True, is_space=True, grid=mock_grid)


def test_resample(mock_signal):
    """Test resampling functionality."""
    original_fs = mock_signal.fs
    target_fs = 24000

    mock_signal.resample(target_fs)

    assert mock_signal.fs == target_fs
    assert mock_signal.n_samples == 500  # Half the samples
    assert mock_signal.data.shape[2] == 500


def test_rotate_space_domain(mock_signal):
    """Test rotating the spatial grid."""
    # Original azimuths: [0, pi/2, pi, 3pi/2]
    orig_az = mock_signal.grid.az.copy()

    # Rotate by 90 degrees around Z axis
    rot = Rotation.from_euler("z", 90, degrees=True)
    mock_signal.rotate_space_domain(rot)

    # New azimuths should be shifted by pi/2
    # [pi/2, pi, 3pi/2, 0] (modulo 2pi)
    expected_az = np.mod(orig_az + np.pi / 2, 2 * np.pi)

    # Check if new azimuths match expected (robust to 0 vs 2pi wrapping)
    # Compare phasors: exp(1j * az)
    actual_phasors = np.exp(1j * mock_signal.grid.az)
    expected_phasors = np.exp(1j * expected_az)

    np.testing.assert_allclose(actual_phasors, expected_phasors, atol=1e-5)

    # Data should remain unchanged
    assert np.allclose(mock_signal.data[0, :, 0], 1.0)


def test_convolve_sh():
    """Test SH convolution logic."""
    fs = 48000
    n_sh = 4  # (1+1)^2
    n_samples = 10

    # Signal 1: Ambisonics (SH, 1, Time)
    # Impulse at t=0 for channel 0
    s1_data = np.zeros((1, n_sh, n_samples))
    s1_data[0, 0, 0] = 1.0
    s1 = SpatialSignal(s1_data, fs, is_time=True, is_space=False, grid=None)

    # Signal 2: HRTF (Ears, SH, Time)
    # Impulse at t=1 for channel 0, left ear
    s2_data = np.zeros((2, n_sh, n_samples))
    s2_data[0, 0, 1] = 1.0  # Left ear
    s2_data[1, 0, 2] = 1.0  # Right ear, t=2
    s2 = SpatialSignal(s2_data, fs, is_time=True, is_space=False, grid=None)

    # Convolve
    output = s1.convolve_sh(s2)

    # Expected:
    # Left ear: Impulse at t=0+1 = 1
    # Right ear: Impulse at t=0+2 = 2

    # output shape is (N1, N2, Time) -> (1, 2, Time)
    assert output.shape == (1, 2, n_samples + n_samples - 1)

    # Check peaks
    # output[0, 0] is Left Ear
    # output[0, 1] is Right Ear
    assert np.argmax(output[0, 0]) == 1
    assert np.argmax(output[0, 1]) == 2
