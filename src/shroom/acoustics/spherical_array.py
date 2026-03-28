import numpy as np
from shroom.geometry.sampling import sphereicalGrid
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.acoustics.physics import (
    _compute_bn_diagonal,
    SPEED_OF_SOUND,
    DEFAULT_SOURCE_DISTANCE,
)
from shroom.utils.dsp_utils import reconstruct_neg_frequency_spectrum
from typing import Optional, Tuple


class SphericalArray(SpatialSignal):
    """
    Frequency-domain steering matrix for a spherical microphone array.

    Models the transfer function between a spherical source grid and
    microphone positions on a rigid or open sphere using modal radial
    functions (Bn).  Inherits from SpatialSignal with shape
    ``(n_mics, n_source_grid, n_samples_or_freqs)``.

    Use ``toSH(N)`` to project the steering matrix to the SH domain,
    enabling efficient SH-domain convolution with an Ambisonics signal.
    """

    def __init__(
        self,
        fs: int,
        duration: np.float32,
        r_sphere: np.float32,
        r_mics: np.ndarray,
        source_grid: sphereicalGrid,
        mics_grid: sphereicalGrid,
        sphere_type: Optional[str] = "rigid",
        sh_order_for_sm_calc: Optional[int] = 14,
        convert_to_time: bool = True,
        source_type: str = "point_source",
        r_s: float = DEFAULT_SOURCE_DISTANCE,
        apply_damping: bool = True,
        normalize_columns: bool = False,
    ):
        """
        Initializes a Spherical Microphone Array simulation.

        Parameters:
        - fs: Sampling frequency in Hz.
        - duration: Duration of the signal in seconds.
        - r_sphere: Radius of the spherical array in meters.
        - r_mics: Array of radii for each microphone in meters. Must match the number of points in mics_grid.
        - source_grid: SamplingGrid object defining the positions of the sound sources.
        - mics_grid: SamplingGrid object defining the positions of the microphones on the sphere.
        - sphere_type: Type of the sphere, either 'rigid' (default) or 'open'.
        - sh_order_for_sm_calc: Maximum order of Spherical Harmonics used for the steering matrix calculation (default is 14).
        - convert_to_time: Object initialized in freq domain, for consistency it is recommended to convert to time.
        - source_type: Source model for the radial functions. 'point_source' (default) uses
          the full Green's function expansion with a point source at distance r_s.
          'plane_wave' uses Bn = 4π·iⁿ·b_n(k), independent of source distance.
        - r_s: Distance of the point source in metres. Only used when source_type='point_source'.
          Default is 2.0.
        - apply_damping: If True, applies Tikhonov magnitude damping and order masking to Bn
          to improve numerical stability. Default is True.
        - normalize_columns: If True, normalizes each steering vector to unit norm across
          the microphone axis for each (source, frequency) pair. Default is False.
        """
        self._validate_child_inputs(
            fs,
            duration,
            source_grid,
            mics_grid,
            r_sphere,
            r_mics,
            sphere_type,
            sh_order_for_sm_calc,
            source_type,
            r_s,
        )
        data = self._calc_sm(
            fs,
            duration,
            source_grid,
            mics_grid,
            r_sphere,
            r_mics,
            sphere_type,
            sh_order_for_sm_calc,
            source_type,
            r_s,
            apply_damping,
            normalize_columns,
        )
        super().__init__(
            data=data, fs=fs, grid=source_grid, is_time=False, is_space=True
        )
        if convert_to_time:
            self.toTime()

    def _calc_sm(
        self,
        fs: int,
        duration: float,
        source_grid: sphereicalGrid,
        array_grid: sphereicalGrid,
        r_sphere: float,
        r_mics: np.ndarray,
        array_type: str,
        N_sh_for_sm_initialization: int,
        source_type: str,
        r_s: float,
        apply_damping: bool,
        normalize_columns: bool,
    ) -> np.ndarray:
        """
        Calculates the spatial microphone signal (transfer function) for the spherical array.

        Returns:
        - H: The computed transfer function matrix (Full Spectrum).
        """
        # 1. Compute Spherical Harmonics
        Y_source = source_grid.Y(N_sh_for_sm_initialization)
        Y_mics = array_grid.Y(N_sh_for_sm_initialization)

        # 2. Compute Frequencies and Wave Numbers
        freqs, pos_wave_number = self._compute_frequencies(fs, duration)
        n_samples = len(freqs)

        # 3. Compute Radial Functions (Bn)
        Bn_diag = self._compute_bn_diagonal_matrix(
            r_mics,
            N_sh_for_sm_initialization,
            pos_wave_number,
            r_sphere,
            array_type,
            n_samples,
            source_type,
            r_s,
            apply_damping,
        )

        # 4. Compute Transfer Function (H) - Positive Frequencies
        H_pos = self._compute_transfer_function(Y_mics, Bn_diag, Y_source)

        # 4b. Normalize each steering vector to unit norm across the mic dimension.
        # For each (source-direction, frequency) pair, ||V[:, q, f]||_2 = 1.
        if normalize_columns:
            norms = np.linalg.norm(H_pos, axis=0, keepdims=True)  # (1, Q, F_pos)
            H_pos /= np.maximum(norms, 1e-12)

        # Force Nyquist to be real if N is even
        if n_samples % 2 == 0:
            H_pos[..., -1] = H_pos[..., -1].real

        # 5. Reconstruct Negative Frequencies
        H = reconstruct_neg_frequency_spectrum(H_pos, n_samples, freq_axis=-1)

        return H

    def _compute_frequencies(
        self, fs: int, duration: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the frequency bins and positive wave numbers."""
        n_samples = int(duration * fs)
        freqs = np.fft.fftfreq(n_samples, 1 / fs)
        pos_freqs = np.fft.rfftfreq(n_samples, 1 / fs)
        pos_wave_number = 2 * np.pi * pos_freqs / SPEED_OF_SOUND
        return freqs, pos_wave_number

    def _compute_bn_diagonal_matrix(
        self,
        r_mics: np.ndarray,
        N_sh: int,
        pos_wave_number: np.ndarray,
        r_sphere: float,
        array_type: str,
        n_freqs: int,
        source_type: str,
        r_s: float,
        apply_damping: bool,
    ) -> np.ndarray:
        """Computes the diagonal matrix of radial functions (Bn)."""
        Bn_diag = np.empty(
            (len(r_mics), int((N_sh + 1) ** 2), int(len(pos_wave_number))),
            dtype=np.complex128,
        )

        for i, r_mic in enumerate(r_mics):
            Bn_diag[i, :] = _compute_bn_diagonal(
                N=N_sh,
                k=pos_wave_number,
                a=r_sphere,
                r_m=r_mic,
                sphere_type=array_type,
                source_type=source_type,
                r_s=r_s,
                apply_damping=apply_damping,
            )
        return Bn_diag

    def _compute_transfer_function(
        self, Y_mics: np.ndarray, Bn_diag: np.ndarray, Y_source: np.ndarray
    ) -> np.ndarray:
        """
        Computes the transfer function H using matrix multiplication.

        H[mic_idx, source_idx, freq_idx]
        Equivalent to: H = np.einsum('il, ilf, jl -> ijf', Y_mics, Bn_diag, Y_source.conj())
        """
        # Y_mics: (M, L)
        # Bn_diag: (M, L, F)
        # Y_source: (Q, L)

        # term: (M, L, F)
        term = Y_mics[:, :, None] * Bn_diag

        # H: (Q, L) @ (M, L, F) -> (Q, M, F) ? No.
        # We want (M, Q, F).
        # H[m, q, f] = sum_l (Y_mics[m,l] * Bn[m,l,f] * Y_source[q,l]*)

        # term[m, l, f] = Y_mics[m,l] * Bn[m,l,f]
        # We want sum_l (term[m,l,f] * Y_source[q,l]*)

        # Transpose term to (M, F, L)
        term_T = term.transpose(0, 2, 1)  # (M, F, L)

        # Y_source.conj().T -> (L, Q)
        # (M, F, L) @ (L, Q) -> (M, F, Q)
        H = np.matmul(term_T, Y_source.conj().T)

        # Transpose to (M, Q, F)
        H = H.transpose(0, 2, 1)

        return H

    def _validate_child_inputs(
        self,
        fs,
        duration,
        source_grid,
        mics_grid,
        r_sphere,
        r_mics,
        sphere_type,
        N_sh_for_sm_initialization,
        source_type,
        r_s,
    ):
        assert sphere_type in ["rigid", "open"], "array_type must be 'rigid' or 'open'"
        assert source_type in ["plane_wave", "point_source"], \
            "source_type must be 'plane_wave' or 'point_source'"
        assert isinstance(r_sphere, float), "r_sphere must be float"
        assert isinstance(r_mics, np.ndarray), "r_mics must be np.ndarray"
        assert isinstance(
            source_grid, sphereicalGrid
        ), "source_grid must be SamplingGrid"
        assert isinstance(
            mics_grid, sphereicalGrid
        ), "array_grid must be SamplingGrid"
        assert (
            N_sh_for_sm_initialization > 0
        ), "N_sh_for_sm_initialization must be positive"
        assert (
            r_mics.shape[0] == mics_grid.n_points
        ), "r_mics must have same length as mics_grid"
        assert r_mics.min() >= r_sphere, "all mics in r_mics must be greater or equal to r_sphere"
        if source_type == "point_source":
            assert r_s > 0, "r_s must be positive"
            assert r_s > r_sphere, "r_s (source distance) must be outside the sphere"
