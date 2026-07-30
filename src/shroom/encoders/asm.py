from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.utils.math_utils import tikhonov
import numpy as np
from shroom.utils.dsp_utils import convolve_and_sum

_EPS = 1e-12


def calculate_asm_coefficients(
    sm: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """
    Calculate ASM coefficients using the formula: W^H = Y^H * V^H * (V * V^H + eps * I)^-1

    Parameters
    ----------
    sm : np.ndarray
        Steering matrix (V) with shape [M, Q, F].
    Y : np.ndarray
        Spherical harmonic matrix with shape [Q, (N_sh+1)**2].
    eps : float, optional
        Regularization parameter, by default 1e-6.

    Returns
    -------
    np.ndarray
        The ASM filter weights with shape [(N_sh+1)**2, M, F].
    """
    M, Q1, F = sm.shape
    Q2, L = Y.shape

    assert Q1 == Q2, f"Y grid ({Q2}) and sm grid ({Q1}) must match."

    C = np.zeros((L, M, F), dtype=np.complex128)
    for f in range(1, F):
        for nm in range(L):
            C[nm, :, f] = tikhonov(A=sm[:, :, f].conj().T, b=Y[:, nm].T)

    # DC (f=0): only the omnidirectional (n=0, m=0) channel is physical.
    # All higher-order channels must be zero; (0,0) must be real.
    C[1:, :, 0] = 0.0
    C[0, :, 0] = C[0, :, 0].real + 0.0j

    # C[1:, :, 1] = 0.0
    # C[0, :, 1] = C[0, :, 0].real + 0.0j

    # Nyquist (f=F//2, even-length FFT): same constraint — must be real.
    if F % 2 == 0:
        C[1:, :, F // 2] = 0.0
        C[0, :, F // 2] = C[0, :, F // 2].real + 0.0j

    return C.transpose(1, 0, 2)


def linear_spectral_magnitude(
    cnm: np.ndarray, sm: np.ndarray, Y: np.ndarray
) -> np.ndarray:
    """
    Per-(SH-channel, frequency) linear spectral magnitude of an ASM solution:

        xi[nm, f] = ‖cnm[:, nm, f]^H V[:, :, f]‖_2 / ‖Y[:, nm]‖_2

    This is the (unsquared) Linear Spectral Error used as the SE-ASM
    equalization target: xi = 1 means the encoded SH channel carries the same
    spectral energy as the ideal SH pattern.

    Parameters
    ----------
    cnm : np.ndarray
        Encoder coefficients with shape [M, (N_sh+1)**2, F].
    sm : np.ndarray
        Steering matrix (V) with shape [M, Q, F].
    Y : np.ndarray
        Spherical harmonic matrix with shape [Q, (N_sh+1)**2].

    Returns
    -------
    np.ndarray
        Linear spectral magnitude with shape [(N_sh+1)**2, F], real-valued.
    """
    proj = np.einsum("mlf,mqf->lqf", cnm.conj(), sm)  # (L, Q, F)
    y_norm = np.maximum(np.linalg.norm(Y, axis=0), _EPS)  # (L,)
    return np.linalg.norm(proj, axis=1) / y_norm[:, np.newaxis]


def calculate_se_asm_coefficients(sm: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Calculate spectrally-equalized ASM (SE-ASM) coefficients.

    SE-ASM rescales the plain ASM (Tikhonov) solution by a real, per-(SH
    channel, frequency) weight derived from the Linear Spectral Error of the
    unweighted solution:

        w[nm, f] = 1 / xi[nm, f],
        xi[nm, f] = ‖cnm[:, nm, f]^H V[:, :, f]‖_2 / ‖Y[:, nm]‖_2

    The weight restores unit spectral magnitude in SH channels that plain ASM
    attenuates at high frequencies (where the array is spatially aliased and
    the regularized solution collapses towards zero), at the cost of a larger
    complex MSE. The equalization is applied over the whole spectrum. Because
    the weights are real and positive, they leave the phase of the ASM filters
    — and hence the SH-domain symmetry of the encoded signal — untouched.

    Parameters
    ----------
    sm : np.ndarray
        Steering matrix (V) with shape [M, Q, F].
    Y : np.ndarray
        Spherical harmonic matrix with shape [Q, (N_sh+1)**2].

    Returns
    -------
    np.ndarray
        The SE-ASM filter weights with shape [M, (N_sh+1)**2, F].
    """
    C = calculate_asm_coefficients(sm, Y)  # (M, L, F)

    xi = linear_spectral_magnitude(C, sm, Y)  # (L, F)
    # Bins with a zero solution (DC / Nyquist) carry no energy to equalize.
    weights = np.where(xi > _EPS, 1.0 / np.maximum(xi, _EPS), 1.0)  # (L, F)

    return C * weights[np.newaxis, :, :]


class ASM:
    """
    Ambisonics Signal Matching (ASM) encoder.
    Calculates filters to encode microphone array signals into Ambisonics.

    With ``spectrally_equalized=True`` the encoder instead produces
    spectrally-equalized ASM (SE-ASM) filters, see
    :func:`calculate_se_asm_coefficients`.
    """

    def __init__(
        self,
        sh_order: int,
        array: SpatialSignal,
        fs: int = None,
        duration: float = None,
        spectrally_equalized: bool = False,
    ):
        """
        Initialize ASM encoder.

        Parameters
        ----------
        sh_order : int
            Target Ambisonics order.
        array : SpatialSignal
            The microphone array response (Steering Matrix) in Frequency Domain.
        fs : int, optional
            Sampling frequency. Must match array.fs if provided.
        duration : float, optional
            Duration of the filters.
        spectrally_equalized : bool, optional
            If True, equalize the linear spectral magnitude of every SH channel
            over the whole spectrum (SE-ASM) instead of returning the plain ASM
            solution. Default is False.
        """
        self._validate_inputs(sh_order, array, fs, duration)
        self.sh_order = sh_order
        self.array = array
        self.fs = fs
        self.duration = duration
        self.spectrally_equalized = spectrally_equalized

        self._cnm = None

    @property
    def cnm(self) -> SpatialSignal:
        """Lazy-loaded ASM coefficients."""
        if self._cnm is None:
            self._cnm = self.calculate()
        return self._cnm

    def calculate(self) -> SpatialSignal:
        """
        Calculate the ASM coefficients (filters).

        Returns SE-ASM coefficients when ``spectrally_equalized`` is True.

        Returns
        -------
        SpatialSignal
            The calculated coefficients in Frequency Domain.
        """
        sm = self.array.data
        Y = self.array.grid.Y(N_sp=self.sh_order)
        if self.spectrally_equalized:
            asm_coefficients = calculate_se_asm_coefficients(sm, Y)
        else:
            asm_coefficients = calculate_asm_coefficients(sm, Y)
        self._cnm = SpatialSignal(
            data=asm_coefficients, fs=self.fs, is_time=False, is_space=False
        )
        return self._cnm

    def encode_amb(self, mics_signals: np.ndarray) -> SpatialSignal:
        """
        Encode microphone signals into Ambisonics using the calculated ASM filters.

        Parameters
        ----------
        mics_signals : np.ndarray
            Microphone signals in Time Domain. Shape (Time, Mics).

        Returns
        -------
        SpatialSignal
            Encoded Ambisonics signal in Time Domain.
        """
        T1, M1 = mics_signals.shape
        M2, Q, F1 = self.cnm.data.shape
        assert (
            M1 == M2
        ), f"mics_signals number of microphones ({M1}), must match array number of microphones ({M2})"

        x = mics_signals.T[np.newaxis, :, :]
        cnm_copy = self.cnm.copy()
        conj = cnm_copy.data.conj()
        cnm_copy.data = conj
        cnm_copy.toTime()
        cnm_copy = cnm_copy.data
        encoded_amb = convolve_and_sum(x, cnm_copy.transpose(1, 0, 2), "time", "time")

        encoded_amb = SpatialSignal(
            data=encoded_amb, fs=self.fs, is_time=True, is_space=False
        )
        return encoded_amb

    def _validate_inputs(self, sh_order, array, fs, duration):
        assert sh_order > 0, "sh_order must be a positive integer."
        assert isinstance(
            array, SpatialSignal
        ), "array must be a SpatialSignal instance."
        assert array.is_space, "array must be in space domain."
        assert array.is_freq, "array must be in frequency domain"
        if fs is not None:
            assert array.fs == fs, f"fs ({fs}) must match array fs ({array.fs})."
        if duration is not None:
            assert duration > 0, f"duration ({duration}) must be a positive number."
