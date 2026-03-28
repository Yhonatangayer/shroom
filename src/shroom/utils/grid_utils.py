from shroom.geometry.sampling import sphereicalGrid
from typing import Tuple
import numpy as np


def fibonacci_sphere_angles(samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate spherical coordinates (azimuth, colatitude) for points evenly distributed on a sphere
    using the Fibonacci method.

    Parameters
    ----------
    samples : int
        Number of points to generate.

    Returns
    -------
    az : np.ndarray
        Azimuthal angles in radians [0, 2*pi).
    co : np.ndarray
        Colatitude angles in radians [0, pi].
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    ga = 2 * np.pi * (1 - 1 / phi)  # Golden angle

    indices = np.arange(samples)

    # Azimuth (phi in math, az here)
    az = (ga * indices) % (2 * np.pi)

    # Height z linearly spaced in [1, -1] (North to South)
    # z = 1 - (2 * i + 1) / N
    z = 1 - (2 * indices + 1) / samples

    # Colatitude (theta in math, co here)
    # arccos(1) = 0 (North), arccos(-1) = pi (South)
    co = np.arccos(z)

    return az, co


def from_fibonacci_grid(n_points: int, sh_type: str = "complex") -> sphereicalGrid:
    """
    Create a sphereicalGrid using Fibonacci sphere sampling.

    Parameters
    ----------
    n_points : int
        Number of points.
    sh_type : str, optional
        Type of Spherical Harmonics ('real' or 'complex').

    Returns
    -------
    sphereicalGrid
        The generated grid.
    """
    az, co = fibonacci_sphere_angles(n_points)

    # Uniform weights summing to 4pi
    weights = np.full(n_points, 4 * np.pi / n_points)

    return sphereicalGrid(az, co, weights, sh_type=sh_type)


def from_spaudiopy_grid(
    spa_grid: Tuple[np.ndarray, np.ndarray], sh_type: str = "complex"
) -> sphereicalGrid:
    """
    Convert a spaudiopy grid tuple (vecs, weights) to SamplingGrid.

    Parameters
    ----------
    spa_grid : Tuple[np.ndarray, np.ndarray]
        Tuple of (vecs, weights), where vecs is (N, 3) and weights is (N,).
    sh_type : str, optional
        Type of Spherical Harmonics ('real' or 'complex').

    Returns
    -------
    sphereicalGrid
        The converted grid.
    """
    vecs, weights = spa_grid
    return sphereicalGrid.from_cartesian(vecs, weights, sh_type=sh_type)
