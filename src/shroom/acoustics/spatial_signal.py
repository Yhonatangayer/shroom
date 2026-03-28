import numpy as np
import warnings
import copy
from typing import Optional, Union, Tuple
from scipy.signal import resample
from scipy.spatial.transform import Rotation
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.dsp_utils import convolve_and_sum
from shroom.utils.amb_utils import get_tilde_matrix
from shroom.utils.rotation_utils import wigner_d_matrix


class SpatialSignal:
    """
    Class representing a general spatial audio signal.
    Can be in Time or Frequency domain, Space or SH domain.
    """

    def __init__(
        self,
        data: np.ndarray,
        fs: int,
        is_time: bool,
        is_space: bool,
        grid: Optional[sphereicalGrid] = None,
    ):
        """
        Initialize SpatialSignal.

        Parameters
        ----------
        data : np.ndarray
            Signal data.
            Expected shape: (n_channels, n_grid/sh_order, n_samples)
        fs : int
            Sampling frequency in Hz.
        is_time : bool
            Flag indicating if data is in time domain.
        is_space : bool
            Flag indicating if data is in space domain (True) or SH domain (False).
        grid : sphereicalGrid, optional
            Spatial sampling grid associated with channels. Required if is_space=True.
        """
        self._validate_inputs(data, fs, grid, is_time, is_space)
        self.data = np.asarray(data)
        self.fs = fs
        self.grid = grid

        self._is_time = is_time
        self._is_space = is_space
        self._orientation = (
            getattr(self.grid, "orientation", None) if self.grid is not None else None
        )

        self._history = []
        self._log_change_to_history("init")

    def _validate_inputs(self, data, fs, grid, is_time, is_space):
        assert isinstance(data, np.ndarray), "data must be np.ndarray"
        assert data.ndim == 3, "data must be 3D array"
        assert fs > 0, "fs must be positive"
        assert isinstance(is_time, bool), "is_time must be bool"
        assert isinstance(is_space, bool), "is_space must be bool"
        if grid is None:
            assert is_space == False, "is_space is True while no grid was provided"
        else:
            assert is_space == True, "is_space is False while a grid was provided"
            assert data.shape[1] == grid.n_points, "data shape must match grid size"

    @property
    def is_time(self) -> bool:
        """Returns True if signal is in Time Domain."""
        return self._is_time

    @property
    def is_freq(self) -> bool:
        """Returns True if signal is in Frequency Domain."""
        return not self._is_time

    @property
    def is_space(self) -> bool:
        """Returns True if signal represents discrete spatial samples (e.g. mic array)."""
        return self._is_space

    @property
    def is_sh(self) -> bool:
        """Returns True if signal is in Spherical Harmonics Domain."""
        return not self._is_space

    @property
    def n_channels(self) -> int:
        """Number of channels (e.g. ears, sources)."""
        return self.data.shape[0]

    @property
    def sh_order(self) -> Optional[int]:
        """Returns the SH order if in SH domain, else return None."""
        if self.is_sh:
            return int(np.sqrt(self.data.shape[1] - 1))
        return None

    @property
    def n_samples(self) -> int:
        """Number of time samples (if in Time Domain)."""
        if self.is_time:
            return self.data.shape[-1]
        else:
            return 0

    @property
    def duration(self) -> Optional[float]:
        """Duration in seconds (only valid for time domain)."""
        if self.is_time:
            return self.n_samples / self.fs
        return None

    @property
    def orientation(self) -> Tuple:
        """Orientation of the spatial grid (if available)."""
        return self._orientation

    def copy(self):
        """
        Create a deep copy of the SpatialSignal instance.
        """
        grid_copy = copy.deepcopy(self.grid) if self.grid is not None else None

        # Create new instance
        new_sig = SpatialSignal(
            data=self.data.copy(),
            fs=self.fs,
            is_time=self.is_time,
            is_space=self.is_space,
            grid=grid_copy,
        )
        # Copy internal state that isn't in init
        new_sig._orientation = copy.deepcopy(self._orientation)
        new_sig._history = copy.deepcopy(self._history)

        return new_sig

    def zero_pad(self, desired_length: int):
        """
        Zero-pad the signal in the time domain to a desired length.

        Parameters
        ----------
        desired_length : int
            The target length in samples.
        """
        if not self.is_time:
            warnings.warn(
                "Zero padding is only supported in time domain. Converting to time domain first."
            )
            self.toTime()

        current_length = self.data.shape[-1]
        n_zero_pad = desired_length - current_length

        if n_zero_pad < 0:
            raise ValueError(
                f"Zero padding desired length({desired_length}) must be greater than current length ({current_length})"
            )

        padding = np.zeros((*self.data.shape[:2], n_zero_pad))
        self.data = np.concatenate((self.data, padding), axis=-1)
        self._log_change_to_history('zero_pad')

    def resample(self, desired_fs: Union[float, np.float32]):
        """
        Resample the signal to a desired sampling frequency.

        Parameters
        ----------
        desired_fs : float
            Target sampling frequency in Hz.
        """
        if not self.is_time:
            warnings.warn(
                "Resampling is only supported in time domain. Converting to time domain first."
            )
            self.toTime()

        if self.fs == desired_fs:
            return

        # Calculate new number of samples
        ratio = desired_fs / self.fs
        new_n_samples = int(self.n_samples * ratio)

        # Resample along time axis (axis=2)
        self.data = resample(self.data, new_n_samples, axis=2)

        # Update fs
        old_fs = self.fs
        self.fs = desired_fs

        self._log_change_to_history(
            "resample", {"old_fs": old_fs, "new_fs": desired_fs}
        )

    def convolve_sh(
        self,
        other: "SpatialSignal",
        sh_order: Optional[int] = None,
        with_tilde: bool = False,
    ) -> np.ndarray:
        """
        Convolve two SH-domain signals.

        Parameters
        ----------
        other : SpatialSignal
            The other signal to convolve with (e.g. HRTF or Array response).
        sh_order : int, optional
            The SH order to use for convolution. Defaults to min of both signals.
        with_tilde : bool, optional
            If True, applies the Tilde matrix (SH normalization) to 'other' before convolution.
            Useful for decoding Ambisonics.

        Returns
        -------
        output : np.ndarray
            Convolved signal in Time Domain. Shape (N1, N2, T1 + T2 - 1).
        """
        # 1. Validation
        if self.is_space or other.is_space:
            raise ValueError("Both signals must be in SH domain (is_space=False).")

        # 2. Ensure Time Domain
        # We work on copies to avoid modifying the originals
        sig1 = self.data if self.is_time else np.fft.ifft(self.data, axis=2)
        sig2 = other.data if other.is_time else np.fft.ifft(other.data, axis=2)

        # 3. Ensure same SH order
        sh_order = sh_order if sh_order else min(self.sh_order, other.sh_order)
        if sh_order is not None:
            sig1 = sig1[:, : (sh_order + 1) ** 2, :]
            sig2 = sig2[:, : (sh_order + 1) ** 2, :]

        # 4. Tilde matrix
        if with_tilde:
            tilde = get_tilde_matrix(sh_order)
            sig2 = np.matmul(tilde, sig2)

        # 5. Convolve and Sum
        out = convolve_and_sum(sig1, sig2, "time", "time")
        return out

    def toFreq(self, nfft: int = None):
        """Convert signal to frequency domain."""
        if self.is_freq:
            warnings.warn("Signal is already in frequency domain.")
            return
        else:
            self.data = np.fft.fft(self.data, axis=2, n=nfft)
            self._is_time = False
            self._log_change_to_history("toFreq", {"nfft": nfft})
            return

    def toTime(self):
        """Convert signal to time domain."""
        if self.is_time:
            warnings.warn("Signal is already in time domain.")
            return
        else:
            self.data = np.fft.ifft(self.data, axis=2)
            self._is_time = True
            self._log_change_to_history("toTime")
            return

    def toSH(self, N_sp):
        """
        Convert signal to Spherical Harmonics domain.

        Parameters
        ----------
        N_sp : int
            Target SH order.
        """
        if self.is_sh:
            warnings.warn("Signal is already in SH domain.")
            return
        else:
            pinvY = self.grid.pinvY(N_sp)
            self.data = np.matmul(pinvY, self.data)
            self._is_space = False
            self.grid = None
            self._log_change_to_history("toSH", {"N_sp": N_sp})
            return

    def toSpace(self, grid: Optional[sphereicalGrid] = None):
        """
        Convert signal to Space domain (reconstruct on a grid).

        Parameters
        ----------
        grid : sphereicalGrid, optional
            Target grid. If None, uses the existing grid (if available) or raises error.
        """
        if self.is_space:
            warnings.warn("Signal is already in space domain.")
            return
        else:
            if grid is not None:
                self.grid = grid
            Y = self.grid.Y(int(np.sqrt(self.data.shape[1]) - 1))
            self.data = np.matmul(Y, self.data)
            self._is_space = True
            self._log_change_to_history(
                "toSpace",
                {
                    "grid n_points": self.grid.n_points,
                },
            )
            return

    def rotate_space_domain(self, rot: Rotation):
        """
        Rotate the spatial grid of the signal.
        Only applicable if is_space=True.

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            Rotation object to apply to the grid.
        """
        if not self.is_space:
            raise ValueError("Signal must be in space domain to rotate grid.")

        if self.grid is None:
            raise ValueError("No grid defined for this signal.")

        self.grid.rotate(rot)

        # update orientation
        if self._orientation is not None:
            self._orientation = rot.apply(self._orientation)

        self._log_change_to_history(
            "rotate_space_domain", {"rot": rot.as_euler("zyx", degrees=True)}
        )

    def rotate_sh_domain(self, rot: Rotation):
        """
        Rotate the signal in Spherical Harmonics domain using Wigner-D matrices.
        Only applicable if is_sh=True.

        Parameters
        ----------
        rot : scipy.spatial.transform.Rotation
            Rotation object.
        """
        if not self.is_sh:
            raise ValueError("Signal must be in SH domain to use rotate_sh_domain.")

        # 1. Get Euler angles (Z-Y-Z convention for Wigner-D)
        # Note: scipy uses intrinsic rotations by default for 'zyz'
        alpha, beta, gamma = rot.as_euler("zyz")

        # 2. Compute Wigner-D matrix
        N = self.sh_order
        D = wigner_d_matrix(N, alpha, beta, gamma)

        # 3. Apply rotation
        # data is (Channels, SH, Time/Freq)
        # D is (SH, SH)
        # We want D @ data
        # einsum: ij, cjk -> cik
        self.data = np.einsum("ij, cjk -> cik", D, self.data)

        # 4. Update orientation
        # Even if we don't have a grid, we track orientation
        if self._orientation is None:
            self._orientation = np.array([1.0, 0.0, 0.0])  # Default forward

        # Rotate orientation vector
        # Orientation is usually a vector in space
        self._orientation = rot.apply(self._orientation)

        self._log_change_to_history(
            "rotate_sh_domain", {"rot": rot.as_euler("zyx", degrees=True)}
        )

    def _log_change_to_history(self, operation_name, params=None):
        """Internal helper to capture the state of the data."""
        domain = ""
        domain += "space X " if self.is_space else "sh X "
        domain += "time" if self.is_time else "freq"
        entry = {
            "operation": operation_name,
            "params": params,
            "shape": self.data.shape,
            "dtype": self.data.dtype,
            "domain": domain,
            "description": f"Object state: {self.data.shape}",
        }
        self._history.append(entry)

    def print_history(self):
        """Print the transformation history of the signal."""
        print("\n--- OBJECT TRANSFORMATION HISTORY ---")
        # Define column widths
        col_widths = {
            "operation": 20,
            "params": 30,
            "shape": 20,
            "dtype": 15,
            "domain": 15,
            "description": 30,
        }

        # Print header
        header = "".join(f"{key:<{width}}" for key, width in col_widths.items())
        print(header)
        print("-" * len(header))

        # Print rows
        for entry in self._history:
            row = ""
            for key, width in col_widths.items():
                val = str(entry.get(key, ""))
                # Truncate if too long
                if len(val) > width - 1:
                    val = val[: width - 4] + "..."
                row += f"{val:<{width}}"
            print(row)

    def __repr__(self):
        domain = "Time" if self.is_time else "Freq"
        space_type = "SH" if self.is_sh else ("Space" if self.is_space else "Channel")
        grid_info = f", grid={self.grid.n_points}pts" if self.grid else ""
        return f"SpatialSignal({domain}, {space_type}, shape={self.data.shape}, fs={self.fs}Hz{grid_info})"
