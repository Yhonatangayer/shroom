import numpy as np
from typing import Tuple, Optional, Union
from scipy.spatial.transform import Rotation
from shroom.utils.math_utils import regularized_pinv
from shroom.utils.amb_utils import sh_matrix


class sphereicalGrid:
    """
    Class representing a spherical sampling grid.
    """

    def __init__(
        self,
        az: np.ndarray,
        co: np.ndarray,
        weights: Optional[np.ndarray] = None,
        orientation: Optional[Union[Tuple[float, float, float], np.ndarray]] = None,
        sh_type: str = "complex",
    ):
        """
        Initialize sphereicalGrid.

        Parameters
        ----------
        az : np.ndarray
            Azimuth in radians, shape (N,).
        co : np.ndarray
            Colatitude in radians, shape (N,).
        weights : np.ndarray, optional
            Quadrature weights, shape (N,).
        orientation : Tuple or np.ndarray, optional
            Orientation vector (x, y, z) or Euler angles. Default is None.
        sh_type : str, optional
            Type of Spherical Harmonics ('real' or 'complex'), by default 'real'.
        """
        # Normalize azimuth to [0, 2pi]
        az = np.mod(az, 2 * np.pi)

        self._validate_inputs(az, co, weights, sh_type)

        # Clip colatitude to [0, pi] to handle floating point errors (after validation)
        self.az = np.asarray(az)
        self.co = np.clip(np.asarray(co), 0, np.pi)

        self.n_points = self.az.shape[0]
        self.sh_type = sh_type

        if orientation is not None:
            self.orientation = np.asarray(orientation)
        else:
            self.orientation = None

        if weights is not None:
            self.weights = np.asarray(weights)
        else:
            self.weights = np.ones(self.n_points) / self.n_points

        # Calculate Cartesian vectors for convenience
        # Standard physics convention:
        # x = r * sin(theta) * cos(phi)
        # y = r * sin(theta) * sin(phi)
        # z = r * cos(theta)
        # where theta is colatitude (0 at North Pole)

        xy_proj = np.sin(self.co)
        x = xy_proj * np.cos(self.az)
        y = xy_proj * np.sin(self.az)
        z = np.cos(self.co)
        self.vecs = np.stack([x, y, z], axis=-1)

        self._Y = None
        self._pinvY = None

    @classmethod
    def from_cartesian(
        cls,
        vecs: np.ndarray,
        weights: Optional[np.ndarray] = None,
        sh_type: str = "complex",
    ):
        """
        Create sphereicalGrid from Cartesian vectors.

        Parameters
        ----------
        vecs : np.ndarray
            Cartesian vectors (N, 3).
        weights : np.ndarray, optional
            Quadrature weights (N,).
        sh_type : str, optional
            Type of Spherical Harmonics ('real' or 'complex').
        """
        vecs = np.asarray(vecs)

        # Normalize vectors to unit sphere for angle calculation
        norms = np.linalg.norm(vecs, axis=1)
        norms[norms < 1e-15] = 1.0
        vecs_norm = vecs / norms[:, np.newaxis]

        # Convert Cartesian to Spherical coordinates
        # az: (-pi, pi], el: [-pi/2, pi/2]
        az = np.arctan2(vecs_norm[:, 1], vecs_norm[:, 0])

        # Handle numerical noise for z
        z = vecs_norm[:, 2]
        z = np.clip(z, -1.0, 1.0)
        co = np.arccos(z)

        grid = cls(az, co, weights, sh_type=sh_type)
        return grid

    def rotate(self, rot: Rotation):
        """
        Rotate the grid points.

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            The rotation to apply.
        """
        # Rotate Cartesian vectors
        self.vecs = rot.apply(self.vecs)

        # Recalculate spherical coordinates
        # Ensure unit vectors for arccos stability
        norms = np.linalg.norm(self.vecs, axis=1)
        # Avoid division by zero
        norms[norms < 1e-15] = 1.0

        x, y, z = self.vecs[:, 0], self.vecs[:, 1], self.vecs[:, 2]

        # Calculate azimuth and normalize to [0, 2pi]
        self.az = np.arctan2(y, x)
        self.az = np.mod(self.az, 2 * np.pi)

        self.co = np.arccos(np.clip(z / norms, -1.0, 1.0))

        # Apply rotation to orientation if provided
        if self.orientation is not None:
            # Assuming orientation is a vector or set of vectors
            self.orientation = rot.apply(self.orientation)

        # Invalidate cached SH matrix
        self._Y = None
        self._pinvY = None

    def Y(self, N_sp: int) -> np.ndarray:
        """
        Compute the Spherical Harmonics Matrix (Y) for this grid.

        Parameters
        ----------
        N_sp : int
            Maximum SH order.

        Returns
        -------
        Y : np.ndarray
            SH matrix of shape (n_points, (N_sp+1)**2).
        """
        if self._Y is None:
            self._Y = self.calculate_Y(N_sp)
        elif self._Y.shape[1] < (N_sp + 1) ** 2:
            self._Y = self.calculate_Y(N_sp)
        return self._Y[:, : (N_sp + 1) ** 2]

    def pinvY(self, N_sp: int) -> np.ndarray:
        """
        Compute the Pseudo-inverse of the Spherical Harmonics Matrix.
        Uses quadrature weights when available for near-exact inversion:
          pinvY = (Y^H W Y)^{-1} Y^H W
        Falls back to regularized pseudo-inverse when no weights are available.

        Parameters
        ----------
        N_sp : int
            Maximum SH order.

        Returns
        -------
        pinvY : np.ndarray
            Pseudo-inverse matrix of shape ((N_sp+1)**2, n_points).
        """
        if self._pinvY is None:
            self._pinvY = self._compute_pinvY(N_sp)
        elif self._pinvY.shape[0] != (N_sp + 1) ** 2:
            self._pinvY = self._compute_pinvY(N_sp)
        return self._pinvY

    def _compute_pinvY(self, N_sp: int) -> np.ndarray:
        Y = self.Y(N_sp)
        # Weighted least-squares: pinvY = (Y^H W Y)^{-1} Y^H W
        # This uses the quadrature weights for near-exact SH analysis on
        # well-sampled grids (avoids regularization bias).
        try:
            YtW = Y.conj().T * self.weights[np.newaxis, :]
            YtWY = YtW @ Y
            return np.linalg.solve(YtWY, YtW)
        except np.linalg.LinAlgError:
            return regularized_pinv(Y)

    def calculate_Y(self, N_sp):
        """
        Internal helper to calculate Y matrix.
        """
        return sh_matrix(N_sp, self.az, self.co, self.sh_type)

    def _validate_inputs(self, az, co, weights, sh_type):
        assert az.shape == co.shape, "azimuth and elevation must have same shape."

        # Check bounds with tolerance for floating point errors
        tol = 1e-5
        if co.min() < -tol or co.max() > np.pi + tol:
            raise ValueError(
                f"elevation must be in [0, pi], currently it is between [{co.min()}, {co.max()}]"
            )

        if weights is not None:
            assert (
                az.shape[0] == weights.shape[0]
            ), "azimuth and weights must have same length."
        assert (
            sh_type == "complex" or sh_type == "real"
        ), "sh_type must be 'complex' or 'real'."

        if sh_type != "complex":
            raise ValueError(
                "currently supports only 'complex' SH type, future versions will support 'real' as well."
            )
