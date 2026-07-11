"""Tests for the low-level spherical-harmonics helpers in shroom.utils.amb_utils."""
import numpy as np
import pytest

from shroom.utils.amb_utils import sh_matrix, get_tilde_matrix
from shroom.utils.grid_utils import from_fibonacci_grid


@pytest.fixture(scope="module")
def dense_grid():
    """A well-sampled Fibonacci grid for approximate quadrature checks."""
    return from_fibonacci_grid(400)


@pytest.mark.parametrize("sh_order", [1, 2, 3])
def test_sh_matrix_complex_shape_and_dtype(sh_order, dense_grid):
    Y = sh_matrix(sh_order, dense_grid.az, dense_grid.co, sh_type="complex")
    assert Y.shape == (dense_grid.n_points, (sh_order + 1) ** 2)
    assert Y.dtype == np.complex128


@pytest.mark.parametrize("sh_order", [1, 2, 3])
def test_sh_matrix_real_shape_and_dtype(sh_order, dense_grid):
    Y = sh_matrix(sh_order, dense_grid.az, dense_grid.co, sh_type="real")
    assert Y.shape == (dense_grid.n_points, (sh_order + 1) ** 2)
    assert Y.dtype == np.float64


def test_real_sh_dc_is_constant(dense_grid):
    """The (0,0) real SH is the constant 1 / (2*sqrt(pi)) everywhere."""
    Y = sh_matrix(3, dense_grid.az, dense_grid.co, sh_type="real")
    np.testing.assert_allclose(Y[:, 0], 1.0 / (2 * np.sqrt(np.pi)), atol=1e-12)


@pytest.mark.parametrize("sh_type", ["real", "complex"])
def test_sh_matrix_orthonormal_on_quadrature_grid(sh_type, dense_grid):
    """Y^H W Y ~= I with uniform quadrature weights on a well-sampled grid."""
    N = 3
    Y = sh_matrix(N, dense_grid.az, dense_grid.co, sh_type=sh_type)
    W = 4 * np.pi / dense_grid.n_points
    gram = (Y.conj().T @ Y) * W
    np.testing.assert_allclose(gram, np.eye((N + 1) ** 2), atol=5e-3)


@pytest.mark.parametrize("sh_order", [0, 1, 2, 3])
def test_tilde_matrix_shape(sh_order):
    T = get_tilde_matrix(sh_order)
    assert T.shape == ((sh_order + 1) ** 2, (sh_order + 1) ** 2)


@pytest.mark.parametrize("sh_order", [1, 2, 3])
def test_tilde_matrix_is_involution(sh_order):
    """The conjugation-permutation matrix is its own inverse: T @ T == I."""
    T = get_tilde_matrix(sh_order)
    np.testing.assert_allclose(T @ T, np.eye((sh_order + 1) ** 2), atol=1e-12)


def test_tilde_matrix_signs():
    """T maps (n, m) -> (n, -m) with sign (-1)^m."""
    T = get_tilde_matrix(1)
    # ACN ordering for N=1: [ (0,0), (1,-1), (1,0), (1,1) ]
    # (1,-1) -> (1,1) with (-1)^{-1} = -1
    assert T[1, 3] == -1
    # (1,0) -> (1,0) with (-1)^0 = +1
    assert T[2, 2] == 1
    # (1,1) -> (1,-1) with (-1)^1 = -1
    assert T[3, 1] == -1
