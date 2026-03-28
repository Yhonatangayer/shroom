import pytest
import numpy as np
from scipy.spatial.transform import Rotation
from shroom.utils.rotation_utils import wigner_d_matrix


def test_wigner_d_properties():
    """
    Test fundamental properties of the Wigner-D matrix implementation.
    """
    N = 3

    # Random rotation
    rot = Rotation.random()
    alpha, beta, gamma = rot.as_euler("zyz")

    # Local implementation (Complex SH)
    D_local = wigner_d_matrix(N, alpha, beta, gamma)

    # 1. Unitary property: D @ D.H = I
    # This ensures energy conservation
    I = np.eye((N + 1) ** 2)
    D_D_H = D_local @ D_local.conj().T

    assert np.allclose(D_D_H, I, atol=1e-10)

    # 2. Determinant should be 1 (magnitude)
    det = np.linalg.det(D_local)
    assert np.isclose(np.abs(det), 1.0)


def test_rotation_consistency():
    """
    Test that rotating by 0 gives Identity, and rotating back gives Identity.
    """
    N = 2

    # Identity
    D_0 = wigner_d_matrix(N, 0, 0, 0)
    assert np.allclose(D_0, np.eye((N + 1) ** 2))

    # Inverse
    # For ZYZ Euler angles (a, b, g), the inverse is (-g, -b, -a)
    alpha, beta, gamma = 0.5, 1.0, -0.5
    D = wigner_d_matrix(N, alpha, beta, gamma)

    D_inv = wigner_d_matrix(N, -gamma, -beta, -alpha)

    # D @ D_inv should be I
    res = D @ D_inv
    assert np.allclose(res, np.eye((N + 1) ** 2), atol=1e-10)


def test_wigner_d_vs_sympy():
    """
    Compare local Wigner-D implementation against SymPy (Reference).
    Tests multiple random angles.
    """
    try:
        from sympy.physics.quantum.spin import Rotation as SymRotation
        from sympy import N as symN
    except ImportError:
        pytest.skip("sympy not installed")

    N_sh = 1  # Keep order low for speed

    # Test multiple random angles
    np.random.seed(42)
    for _ in range(5):
        alpha, beta, gamma = np.random.uniform(0, 2 * np.pi, 3)

        # Local
        D_local = wigner_d_matrix(N_sh, alpha, beta, gamma)

        # SymPy
        n = 1
        start_idx = n**2
        D_block_local = D_local[
            start_idx : start_idx + (2 * n + 1), start_idx : start_idx + (2 * n + 1)
        ]

        D_block_sympy = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex128)

        for i, m_row in enumerate(range(-n, n + 1)):  # m'
            for j, m_col in enumerate(range(-n, n + 1)):  # m
                val = SymRotation.D(n, m_row, m_col, alpha, beta, gamma).doit()
                D_block_sympy[i, j] = complex(val)

        # Compare Magnitudes
        assert np.allclose(np.abs(D_block_local), np.abs(D_block_sympy), atol=1e-5)

        # Check consistency up to sign flips
        diff = np.abs(D_block_local - D_block_sympy)
        sum_diff = np.abs(D_block_local + D_block_sympy)

        is_match = np.logical_or(diff < 1e-5, sum_diff < 1e-5)
        assert np.all(is_match), f"Mismatch for angles {alpha, beta, gamma}"


def test_specific_rotations_vs_sympy():
    """
    Test specific rotations (Yaw, Pitch, Roll) including Up/Down against SymPy.
    """
    try:
        from sympy.physics.quantum.spin import Rotation as SymRotation
    except ImportError:
        pytest.skip("sympy not installed")

    N_sh = 1

    # Define 6 test cases (Euler ZYZ angles)
    rotations = [
        ("Identity", [0, 0, 0]),
        ("Yaw 90 (Left)", [45, 0, 0]),
        ("Yaw -90 (Right)", [-45, 0, 0]),
        ("Pitch 90 (Up)", [0, 45, 0]),
        ("Pitch -90 (Down)", [0, -45, 0]),
        ("Roll 90", [0, 0, 45]),
    ]

    for name, angles_deg in rotations:
        # Convert to ZYZ
        rot = Rotation.from_euler("zyx", angles_deg, degrees=True)
        alpha, beta, gamma = rot.as_euler("zyz")

        # Local
        D_local = wigner_d_matrix(N_sh, alpha, beta, gamma)

        # SymPy
        n = 1
        start_idx = n**2
        D_block_local = D_local[
            start_idx : start_idx + (2 * n + 1), start_idx : start_idx + (2 * n + 1)
        ]

        D_block_sympy = np.zeros((2 * n + 1, 2 * n + 1), dtype=np.complex128)

        for i, m_row in enumerate(range(-n, n + 1)):
            for j, m_col in enumerate(range(-n, n + 1)):
                val = SymRotation.D(n, m_row, m_col, alpha, beta, gamma).doit()
                D_block_sympy[i, j] = complex(val)

        # Compare Magnitudes
        assert np.allclose(
            np.abs(D_block_local), np.abs(D_block_sympy), atol=1e-5
        ), f"Magnitude mismatch for {name}"

        # Check consistency up to sign flips
        diff = np.abs(D_block_local - D_block_sympy)
        sum_diff = np.abs(D_block_local + D_block_sympy)

        is_match = np.logical_or(diff < 1e-5, sum_diff < 1e-5)
        assert np.all(is_match), f"Sign/Value mismatch for {name}"
