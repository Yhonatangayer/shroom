import pytest
import numpy as np
import os
from shroom.utils.grid_utils import from_fibonacci_grid, fibonacci_sphere_angles
from shroom.geometry.sampling import sphereicalGrid
# from spaudiopy.grids import load_lebedev
from shroom.utils.math_utils import regularized_pinv
from shroom.utils.file_utils import load_file
from shroom.paths import DEFAULT_HRTF_PATH

# Path to HRTF file
HRTF_PATH = DEFAULT_HRTF_PATH


@pytest.fixture
def lebedev_grid_low():
    """Lebedev grid degree 11 (50 points). Sufficient for N=3."""
    return from_fibonacci_grid(50)


@pytest.fixture
def lebedev_grid_high():
    """Lebedev grid degree 29 (170 points). Sufficient for N=10."""
    return from_fibonacci_grid(170)


def test_pinvY_low_order(lebedev_grid_low):
    """Test pinvY for low SH order (well-conditioned)."""
    N_sh = 3
    L = (N_sh + 1) ** 2

    Y = lebedev_grid_low.Y(N_sh)
    pinvY = lebedev_grid_low.pinvY(N_sh)

    # Check dimensions
    assert Y.shape == (50, L)
    assert pinvY.shape == (L, 50)

    # Check reconstruction: pinvY @ Y should be Identity (L x L)
    recon = np.matmul(pinvY, Y)
    identity = np.eye(L)

    # Allow some tolerance
    np.testing.assert_allclose(recon, identity, atol=2e-4)


def test_pinvY_high_order_stability(lebedev_grid_high):
    """Test pinvY for high SH order (checking stability)."""
    N_sh = 10
    L = (N_sh + 1) ** 2  # 121

    Y = lebedev_grid_high.Y(N_sh)
    pinvY = lebedev_grid_high.pinvY(N_sh)

    # Check reconstruction
    recon = np.matmul(pinvY, Y)
    identity = np.eye(L)

    # Relax tolerance for high order
    np.testing.assert_allclose(recon, identity, atol=0.1)
    assert np.max(np.abs(pinvY)) < 100.0


def test_round_trip_partial_grid():
    """
    Test Space -> SH -> Space round trip on a partial grid (e.g. Hemisphere).
    This simulates HRTF processing where the grid is often incomplete.
    """
    # Create a grid on the upper hemisphere only
    full_grid = fibonacci_sphere_angles(50)
    azimuth = full_grid[0]
    colatitude = full_grid[1]

    mask = colatitude<(np.pi/2)  # Upper hemisphere
    hemi_azimuth = azimuth[mask]
    hemi_colatitude = colatitude[mask]
    grid = sphereicalGrid(
        az = hemi_azimuth,
        co = hemi_colatitude,
    )

    N_sh = 3

    # Create a synthetic signal on this grid
    # Let's assume the signal is a simple Y_1,0 (dipole pointing up)
    # This should be perfectly reconstructible as it's in the covered region.

    # 1. Generate signal in Space
    Y_true = grid.Y(N_sh)
    # Signal = Y_1,0 (Index 2 in ACN: 0=00, 1=1-1, 2=10, 3=11)
    signal_space = Y_true[:, 2]

    # 2. Convert to SH (Encoding)
    pinvY = grid.pinvY(N_sh)
    coeffs_sh = pinvY @ signal_space

    # 3. Convert back to Space (Decoding)
    signal_recon = Y_true @ coeffs_sh

    # 4. Check Error
    mse = np.mean(np.abs(signal_space - signal_recon) ** 2)
    print(f"Round Trip MSE: {mse}")

    # Relaxed tolerance due to removal of scaling factor
    assert mse < 2e-6

    # Check that the coefficient for Y_1,0 is significant (e.g. > 0.5)
    print(f"Coeff Y_1,0: {coeffs_sh[2]}")
    assert np.abs(coeffs_sh[2]) > 0.75

    # Check that other coefficients are small (aliasing/leakage should be low)
    coeffs_sh[2] = 0
    assert np.max(np.abs(coeffs_sh)) < 0.25


def test_regularization_with_noise():
    """
    Test that regularized_pinv is more robust to noise on an irregular grid.
    """
    # Create irregular grid (random points)
    n_points = 100
    az = np.random.uniform(0, 2 * np.pi, n_points)
    co = np.random.uniform(0, np.pi, n_points)
    grid = sphereicalGrid(az, co)

    N_sh = 5
    L = (N_sh + 1) ** 2

    Y = grid.Y(N_sh)

    # Signal: Y_0,0 + Noise
    signal_clean = Y[:, 0]
    noise = np.random.randn(n_points) * 0.1
    signal_noisy = signal_clean + noise

    # 1. Standard PINV
    pinv_std = np.linalg.pinv(Y)
    coeffs_std = pinv_std @ signal_noisy

    # 2. Regularized PINV
    pinv_reg = regularized_pinv(Y)
    coeffs_reg = pinv_reg @ signal_noisy

    # Check norm of coefficients (Regularized should be smaller/smoother)
    norm_std = np.linalg.norm(coeffs_std)
    norm_reg = np.linalg.norm(coeffs_reg)

    print(f"Std Norm: {norm_std}, Reg Norm: {norm_reg}")

    # We expect regularization to suppress noise amplification
    assert norm_reg < norm_std


def test_regularization_accuracy_ill_conditioned():
    """
    Test accuracy and stability on a highly ill-conditioned grid.
    """
    # Create a grid concentrated in a very small region (e.g., 5 degrees cap)
    # This makes global SH estimation extremely unstable.
    n_points = 100
    az = np.random.uniform(0, 0.1, n_points)  # Small azimuth range
    co = np.random.uniform(0, 0.1, n_points)  # Small colatitude range (North Pole)
    grid = sphereicalGrid(az, co)

    N_sh = 3  # Even low order is hard if we only see the North Pole
    Y = grid.Y(N_sh)

    # Signal: A mode that is NOT well observed (e.g., Y_1,1 which is 0 at pole)
    # Actually, let's use a random combination
    coeffs_true = np.random.randn(Y.shape[1])
    signal_clean = Y @ coeffs_true

    # Add noise
    noise = np.random.randn(n_points) * 0.01
    signal_noisy = signal_clean + noise

    # 1. Standard PINV
    pinv_std = np.linalg.pinv(Y)
    coeffs_std = pinv_std @ signal_noisy

    # 2. Regularized PINV
    pinv_reg = regularized_pinv(Y)
    coeffs_reg = pinv_reg @ signal_noisy

    # Metrics
    norm_std = np.linalg.norm(coeffs_std)
    norm_reg = np.linalg.norm(coeffs_reg)

    # Reconstruct signal on the grid (should fit data)
    recon_std = Y @ coeffs_std
    recon_reg = Y @ coeffs_reg

    mse_std = np.mean((recon_std - signal_noisy) ** 2)
    mse_reg = np.mean((recon_reg - signal_noisy) ** 2)

    print(f"Ill-Conditioned Grid:")
    print(f"Std Norm: {norm_std:.2f}, MSE: {mse_std:.6f}")
    print(f"Reg Norm: {norm_reg:.2f}, MSE: {mse_reg:.6f}")

    # Expectation:
    # 1. Regularized norm should be MUCH smaller (orders of magnitude)
    assert norm_reg < 0.1 * norm_std

    # 2. MSE should be comparable (regularization shouldn't destroy the fit on the observed region)
    # It might be slightly worse, but not exploded.
    assert mse_reg < 10 * mse_std  # Loose bound, just sanity check


@pytest.mark.skipif(not os.path.exists(HRTF_PATH), reason="HRTF file not found")
def test_hrtf_grid_regularization():
    """
    Test regularization benefit on the actual HRTF grid.
    """
    hrtf = load_file(HRTF_PATH)
    grid = hrtf.grid

    # Use a high order typical for HRTF
    N_sh = 10
    L = (N_sh + 1) ** 2

    # Check if grid has enough points
    if grid.n_points < L:
        pytest.skip(
            f"HRTF grid has {grid.n_points} points, not enough for N={N_sh} ({L} coeffs)"
        )

    Y = grid.Y(N_sh)

    # Create a noisy signal
    # Assume the true field is just Monopole (Y_00)
    signal_clean = Y[:, 0]
    noise = np.random.randn(grid.n_points) * 0.1
    signal_noisy = signal_clean + noise

    # 1. Standard PINV
    pinv_std = np.linalg.pinv(Y)
    coeffs_std = pinv_std @ signal_noisy

    # 2. Regularized PINV
    pinv_reg = regularized_pinv(Y)
    coeffs_reg = pinv_reg @ signal_noisy

    norm_std = np.linalg.norm(coeffs_std)
    norm_reg = np.linalg.norm(coeffs_reg)

    print(f"HRTF Grid - Std Norm: {norm_std:.2f}, Reg Norm: {norm_reg:.2f}")

    # Assert regularization helps
    assert norm_reg < norm_std
