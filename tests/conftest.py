"""Shared pytest fixtures for the shroom test suite.

Factory fixtures for the array/grid setups that were previously duplicated across
test_spherical_array.py, test_asm.py, and test_processors.py. Each returns a builder
so individual tests can override only the parameters they care about.
"""
import numpy as np
import pytest

from shroom.acoustics.spherical_array import SphericalArray
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid


@pytest.fixture
def make_equator_grid():
    """Factory: a ring of ``n_mics`` points on the equator (co = pi/2)."""
    def _make(n_mics: int) -> sphereicalGrid:
        return sphereicalGrid(
            az=np.linspace(0, 2 * np.pi, n_mics, endpoint=False),
            co=np.full(n_mics, np.pi / 2),
        )
    return _make


@pytest.fixture
def make_spherical_array(make_equator_grid):
    """Factory for a rigid ``SphericalArray`` with mics on the equator and a
    Fibonacci source grid. Override any parameter as needed."""
    def _make(
        n_mics: int = 6,
        radius: float = 0.1,
        fs: int = 48000,
        duration: float = 0.01,
        source_points: int = 50,
        sh_order_for_sm_calc: int = 3,
        sphere_type: str = "rigid",
        convert_to_time: bool = False,
    ) -> SphericalArray:
        return SphericalArray(
            fs=fs,
            duration=duration,
            r_sphere=radius,
            r_mics=np.full(n_mics, radius),
            source_grid=from_fibonacci_grid(source_points),
            mics_grid=make_equator_grid(n_mics),
            sphere_type=sphere_type,
            sh_order_for_sm_calc=sh_order_for_sm_calc,
            convert_to_time=convert_to_time,
        )
    return _make
