import numpy as np
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.utils.math_utils import magls, tikhonov
from shroom.utils.dsp_utils import convolve_and_sum
from shroom.utils.dsp_utils import (
    reconstruct_neg_frequency_spectrum,
)


def calculate_bsm_coefficients(
    V: np.ndarray,
    h: np.ndarray,
    fs: int,
    use_magls: bool = False,
    magls_cutoff_frequency: float = 1500.0,
    lam: float = None,
    beta: float = None,
) -> tuple:
    """
    Calculate BSM beamformer weights.

    Solves for each positive frequency f:
        V_f @ c_f^* ≈ h_ear_f^*
    where V_f = V[:, :, f].T  shape (Q, M)

    Parameters
    ----------
    V : np.ndarray, shape (M, Q, F)
        Steering matrix (frequency domain).
    h : np.ndarray, shape (2, Q, F)
        HRTF in space domain (frequency domain).
    fs : int
        Sampling frequency in Hz.
    use_magls : bool
        Whether to apply Magnitude Least Squares above the cutoff frequency.
    magls_cutoff_frequency : float
        Cutoff frequency (Hz) for MagLS crossover.
    lam : float or None
        Explicit Tikhonov regularization parameter. Passed to both the regular
        LS and MagLS solvers. If None and beta is also None, lambda is estimated
        adaptively from the singular values of the steering matrix at each bin.
    beta : float or None
        Convenience regularization parameter: sets lam = beta**2 for both the
        regular LS and MagLS solvers. Matches the MATLAB BSM toolbox convention
        (beta=0.075 → lam=5.625e-3). Overrides lam when provided.

    Returns
    -------
    cl : np.ndarray, shape (F, M)
        Left ear beamformer weights (full spectrum).
    cr : np.ndarray, shape (F, M)
        Right ear beamformer weights (full spectrum).
    """
    if beta is not None:
        lam = beta ** 2

    V = V.astype(np.complex128)
    h = h.astype(np.complex128)
    M, Q, F = V.shape

    assert Q == h.shape[1], f"V grid ({Q}) and h grid ({h.shape[1]}) must match."
    assert F == h.shape[2], f"V ({F}) and h ({h.shape[2]}) must have matching freq bins."

    F_pos = F // 2 + 1

    # Initialize full-spectrum filters
    cl = np.zeros((F_pos, M), dtype=np.complex128)
    cr = np.zeros((F_pos, M), dtype=np.complex128)

    # Compute least-squares solution for positive frequencies
    for f in range(F_pos):
        cl[f, :] = tikhonov(V[:, :, f].conj().T, h[0, :, f].conj())#, lam=lam)
        cr[f, :] = tikhonov(V[:, :, f].conj().T, h[1, :, f].conj())#, lam=lam)

    if use_magls:
        pos_freqs = np.fft.rfftfreq(F, 1.0 / fs)
        f_cutoff_idx = int(np.argmin(np.abs(pos_freqs - magls_cutoff_frequency)))

        for f in range(f_cutoff_idx, F_pos):
            alpha = min(1.0, (f - f_cutoff_idx) / (f_cutoff_idx * (np.sqrt(2) - 1)))

            cl[f, :] = alpha * magls(
                A=V[:, :, f].conj().T,
                b=h[0, :, f].conj(),
                x_prev=cl[f - 1, :],
                A_prev=V[:, :, f - 1].conj().T,
                lam=lam,
            ) + (1 - alpha) * cl[f, :]

            cr[f, :] = alpha * magls(
                A=V[:, :, f].conj().T,
                b=h[1, :, f].conj(),
                x_prev=cr[f - 1, :],
                A_prev=V[:, :, f - 1].conj().T,
                lam=lam,
            ) + (1 - alpha) * cr[f, :]

    # DC and Nyquist: enforce real-valued filters
    cl[0, :] = cl[0, :].real + 0.0j
    cr[0, :] = cr[0, :].real + 0.0j
    if F % 2 == 0:
        cl[F // 2, :] = cl[F // 2, :].real + 0.0j
        cr[F // 2, :] = cr[F // 2, :].real + 0.0j

    # Reconstruct negative frequencies via Hermitian symmetry
    cl = reconstruct_neg_frequency_spectrum(cl, n_fft=F, freq_axis=0)
    cr = reconstruct_neg_frequency_spectrum(cr, n_fft=F, freq_axis=0)

    return cl, cr


class BSM:
    """
    Beamformer-Steered Matching (BSM) encoder.
    Calculates beamformer filters that encode microphone array signals into binaural audio.
    """

    def __init__(
        self,
        array: SpatialSignal,
        hrtf: SpatialSignal,
        use_magls: bool = False,
        magls_cutoff_frequency: float = 1200.0,
        fs: int = None,
        duration: float = None,
        lam: float = None,
        beta: float = None,
    ):
        """
        Initialize BSM encoder.

        Parameters
        ----------
        array : SpatialSignal
            Steering matrix, shape (M, Q, F), is_space=True, is_freq=True.
        hrtf : SpatialSignal
            HRTF in space domain, shape (2, Q, F), is_space=True, is_freq=True.
        use_magls : bool
            Apply Magnitude Least Squares above the cutoff frequency.
        magls_cutoff_frequency : float
            Cutoff frequency (Hz) for MagLS crossover.
        fs : int, optional
            Sampling frequency. Must match array.fs if provided.
        duration : float, optional
            Duration of the filters in seconds.
        lam : float or None
            Explicit Tikhonov regularization parameter (used for both LS and MagLS).
            If None and beta is also None, lambda is estimated adaptively per bin.
        beta : float or None
            Convenience parameter: sets lam = beta**2 (matches MATLAB BSM toolbox
            convention, e.g. beta=0.075). Overrides lam when provided.
        """
        self._validate_inputs(array, hrtf, fs, duration)
        self.array = array
        self.hrtf = hrtf
        self.use_magls = use_magls
        self.magls_cutoff_frequency = magls_cutoff_frequency
        self.fs = array.fs
        self.duration = duration
        self.lam = lam
        self.beta = beta

        self._cl = None
        self._cr = None

    @property
    def cl(self) -> np.ndarray:
        """Left ear beamformer weights, shape (F, M). Computed lazily."""
        if self._cl is None:
            self._cl, self._cr = self._calculate_coefficients()
        return self._cl

    @property
    def cr(self) -> np.ndarray:
        """Right ear beamformer weights, shape (F, M). Computed lazily."""
        if self._cr is None:
            self._cl, self._cr = self._calculate_coefficients()
        return self._cr

    def get_coefficients(self):
        """Return (cl, cr) beamformer weights, each shape (F, M)."""
        return self.cl, self.cr

    def _calculate_coefficients(self):
        return calculate_bsm_coefficients(
            V=self.array.data,
            h=self.hrtf.data,
            fs=self.fs,
            use_magls=self.use_magls,
            magls_cutoff_frequency=self.magls_cutoff_frequency,
            lam=self.lam,
            beta=self.beta,
        )

    def process(self, mic_signals: SpatialSignal) -> SpatialSignal:
        """
        Encode microphone signals to binaural audio using the BSM filters.

        Uses time-domain convolution so the input can have any length.

        Parameters
        ----------
        mic_signals : SpatialSignal
            Microphone signals, shape (M, 1, T), is_time=True.

        Returns
        -------
        SpatialSignal
            Binaural audio (2 channels: L/R), shape (2, 1, T), is_time=True, is_space=False.
        """
        if not mic_signals.is_time:
            raise ValueError("BSM.process expects time-domain input.")

        M, _, T = mic_signals.data.shape
        assert M == self.array.data.shape[0], \
            f"mic_signals mics ({M}) must match array mics ({self.array.data.shape[0]})."

        # Convert beamformer filters to time domain: (F, M) -> (F, M)
        cl_time = np.fft.ifft(self.cl.conj(), axis=0)
        cr_time = np.fft.ifft(self.cr.conj(), axis=0)

        mic_data = mic_signals.data[:, 0, :]  # (M, T)

        # Convolve all mics with both ear filters in a single vectorised call.
        # convolve_and_sum(signal1=(1,M,T), signal2=(2,M,F)) → (1, 2, T+F-1)
        # The M axis is summed over (shared channel), so no Python loop needed.
        x       = mic_data[np.newaxis, :, :]                        # (1, M, T)
        filters = np.stack([cl_time.T, cr_time.T], axis=0)          # (2, M, F)
        out     = convolve_and_sum(x, filters, "time", "time")       # (1, 2, T+F-1)

        y_L = out[0, 0, :T]
        y_R = out[0, 1, :T]

        binaural = np.stack([y_L, y_R], axis=0)[:, np.newaxis, :]  # (2, 1, T)

        return SpatialSignal(
            data=binaural,
            fs=self.fs,
            is_time=True,
            is_space=False,
        )

    def _validate_inputs(self, array, hrtf, fs, duration):
        assert isinstance(array, SpatialSignal), "array must be a SpatialSignal instance."
        assert array.is_space, "array must be in space domain."
        assert array.is_freq, "array must be in frequency domain."
        assert isinstance(hrtf, SpatialSignal), "hrtf must be a SpatialSignal instance."
        assert hrtf.is_space, "hrtf must be in space domain."
        assert hrtf.is_freq, "hrtf must be in frequency domain."
        assert array.data.shape[1] == hrtf.data.shape[1], \
            f"array grid ({array.data.shape[1]}) and hrtf grid ({hrtf.data.shape[1]}) must match."
        assert array.data.shape[2] == hrtf.data.shape[2], \
            f"array ({array.data.shape[2]}) and hrtf ({hrtf.data.shape[2]}) must have matching freq bins."
        if fs is not None:
            assert array.fs == fs, f"fs ({fs}) must match array.fs ({array.fs})."
        if duration is not None:
            assert duration > 0, "duration must be positive."
