import pytest
import numpy as np
from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid


@pytest.fixture
def basic_array():
    """Fixture for a basic spherical array."""
    fs = 48000
    radius = 0.1
    n_mics = 6

    mics_grid = sphereicalGrid(
        az=np.linspace(0, 2 * np.pi, n_mics, endpoint=False),
        co=np.full(n_mics, np.pi / 2),
    )

    source_grid = from_fibonacci_grid(50)

    return SphericalArray(
        fs=fs,
        duration=0.032,
        r_sphere=radius,
        r_mics=np.full(n_mics, radius),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=3,
    )


def test_array_initialization(basic_array):
    """Test array initialization and properties."""
    assert basic_array.fs == 48000
    assert basic_array.n_channels == 6  # 6 mics
    assert basic_array.is_space  # It's a spatial signal (mic signals)
    assert basic_array.grid.n_points == 50  # lebdev with deg=11


def test_array_data_shape(basic_array):
    """Test the shape of the generated array manifold/data."""
    # Data shape should be (n_mics, n_sources, n_samples)
    # But SpatialSignal expects (n_channels, n_grid, n_samples)
    # For SphericalArray, n_channels=n_mics, n_grid=n_sources (from source_grid)

    n_mics = 6
    n_sources = 50  # Lebedev degree 11

    assert basic_array.data.shape[0] == n_mics
    assert basic_array.data.shape[1] == n_sources
    assert basic_array.data.shape[2] > 0  # Time samples
