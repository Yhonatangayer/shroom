import pytest
import numpy as np
from shroom.encoders.asm import (
    ASM,
    calculate_asm_coefficients,
    calculate_se_asm_coefficients,
    linear_spectral_magnitude,
)
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.utils.dsp_utils import is_signal_frequency_sh_valid


@pytest.fixture
def real_array_signal(make_spherical_array):
    """Create a real SphericalArray signal in frequency domain."""
    array = make_spherical_array(
        n_mics=6,
        radius=0.1,
        fs=16000,
        duration=0.01,
        source_points=50,
        sh_order_for_sm_calc=3,
        convert_to_time=True,
    )

    # validation of representation
    array.toFreq()
    array_copy = array.copy()
    array_copy.toSH(N_sp=3)
    assert array_copy.is_sh and array_copy.is_freq
    assert is_signal_frequency_sh_valid(array_copy.data, freq_axis=-1)

    return array


def test_calculate_asm_coefficients(real_array_signal):
    """Test the standalone calculation function."""
    sm = real_array_signal.data
    sh_order = 1
    Y = real_array_signal.grid.Y(N_sp=sh_order)

    # sm: (M, Q, F)
    # Y: (Q, L)

    coeffs = calculate_asm_coefficients(sm, Y)

    # Expected shape: (L, M, F)
    L = (sh_order + 1) ** 2
    M = sm.shape[0]
    F = sm.shape[2]

    assert coeffs.shape == (M, L, F)
    assert coeffs.dtype == np.complex128


def test_asm_class(real_array_signal):
    """Test the ASM class wrapper."""
    sh_order = 1
    asm = ASM(
        sh_order=sh_order,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
    )

    # Test cnm representation
    cnm_signal = asm.calculate()

    assert isinstance(cnm_signal, SpatialSignal)  # Should be SpatialSignal
    assert cnm_signal.is_freq
    assert cnm_signal.is_sh  # SH domain

    L = (sh_order + 1) ** 2
    M = real_array_signal.n_channels
    F = real_array_signal.data.shape[2]

    assert cnm_signal.data.shape == (M, L, F)

def test_asm_dc_constraint(real_array_signal):
    """ASM DC bin: higher-order channels are zero, omnidirectional is real.

    Array-agnostic property of the ASM encoder (also checked against the real ARIA
    array in test_aria_asm_bsm.py, which is skipped when that data file is absent).
    """
    sh_order = 1
    asm = ASM(sh_order=sh_order, array=real_array_signal, fs=real_array_signal.fs, duration=0.1)
    cnm = asm.calculate().data  # (M, nm, F)
    assert np.allclose(cnm[:, 1:, 0], 0.0), "ASM DC: higher-order channels are not zero."
    assert np.allclose(cnm[:, 0, 0].imag, 0.0), "ASM DC: (0,0) channel is not real."


def test_asm_nyquist_constraint(real_array_signal):
    """ASM Nyquist bin: higher-order channels are zero, omnidirectional is real."""
    sh_order = 1
    asm = ASM(sh_order=sh_order, array=real_array_signal, fs=real_array_signal.fs, duration=0.1)
    cnm = asm.calculate().data  # (M, nm, F)
    F = cnm.shape[-1]
    if F % 2 == 0:
        nyq = F // 2
        assert np.allclose(cnm[:, 1:, nyq], 0.0), "ASM Nyquist: higher-order channels are not zero."
        assert np.allclose(cnm[:, 0, nyq].imag, 0.0), "ASM Nyquist: (0,0) channel is not real."


def test_se_asm_shape_and_domain(real_array_signal):
    """SE-ASM returns the same kind of SpatialSignal as plain ASM."""
    sh_order = 1
    se_asm = ASM(
        sh_order=sh_order,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
        spectrally_equalized=True,
    )
    cnm_signal = se_asm.calculate()

    assert isinstance(cnm_signal, SpatialSignal)
    assert cnm_signal.is_freq and cnm_signal.is_sh

    L = (sh_order + 1) ** 2
    M = real_array_signal.n_channels
    F = real_array_signal.data.shape[2]
    assert cnm_signal.data.shape == (M, L, F)


def test_se_asm_differs_from_asm(real_array_signal):
    """The flag must actually change the filters (default stays plain ASM)."""
    kwargs = dict(sh_order=1, array=real_array_signal, fs=real_array_signal.fs, duration=0.1)
    cnm_asm = ASM(**kwargs).cnm.data
    cnm_se = ASM(spectrally_equalized=True, **kwargs).cnm.data

    assert not np.allclose(cnm_asm, cnm_se)
    # Default is plain ASM.
    assert np.allclose(ASM(spectrally_equalized=False, **kwargs).cnm.data, cnm_asm)


def test_se_asm_equalizes_spectral_magnitude(real_array_signal):
    """Every equalized SH channel carries unit linear spectral magnitude."""
    sm = real_array_signal.data
    Y = real_array_signal.grid.Y(N_sp=1)

    cnm_se = calculate_se_asm_coefficients(sm, Y)
    xi = linear_spectral_magnitude(cnm_se, sm, Y)  # (L, F)

    F = sm.shape[2]
    # DC and Nyquist are constrained to zero by construction and are excluded.
    active = np.ones(F, dtype=bool)
    active[0] = False
    if F % 2 == 0:
        active[F // 2] = False

    assert np.allclose(xi[:, active], 1.0, atol=1e-8)


def test_se_asm_preserves_asm_phase(real_array_signal):
    """SE-ASM only rescales the ASM filters — the weights are real & positive."""
    sm = real_array_signal.data
    Y = real_array_signal.grid.Y(N_sp=1)

    cnm_asm = calculate_asm_coefficients(sm, Y)
    cnm_se = calculate_se_asm_coefficients(sm, Y)

    nonzero = np.abs(cnm_asm) > 1e-12
    ratio = cnm_se[nonzero] / cnm_asm[nonzero]
    assert np.allclose(ratio.imag, 0.0)
    assert np.all(ratio.real > 0.0)


def test_se_asm_dc_and_nyquist_constraints(real_array_signal):
    """Equalization preserves the ASM DC/Nyquist constraints."""
    cnm = ASM(
        sh_order=1,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
        spectrally_equalized=True,
    ).cnm.data  # (M, nm, F)

    assert np.allclose(cnm[:, 1:, 0], 0.0), "SE-ASM DC: higher-order channels are not zero."
    assert np.allclose(cnm[:, 0, 0].imag, 0.0), "SE-ASM DC: (0,0) channel is not real."

    F = cnm.shape[-1]
    if F % 2 == 0:
        nyq = F // 2
        assert np.allclose(cnm[:, 1:, nyq], 0.0), "SE-ASM Nyquist: higher-order channels are not zero."
        assert np.allclose(cnm[:, 0, nyq].imag, 0.0), "SE-ASM Nyquist: (0,0) channel is not real."


def test_se_asm_filters_are_real_in_time(real_array_signal):
    """Real, even weights keep the filter spectrum conjugate-symmetric."""
    cnm = ASM(
        sh_order=1,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
        spectrally_equalized=True,
    ).cnm.data
    assert is_signal_frequency_sh_valid(cnm, freq_axis=-1)


def test_encode_amb(real_array_signal):
    """Test encoding microphone signals to Ambisonics."""
    sh_order = 1
    asm = ASM(
        sh_order=sh_order,
        array=real_array_signal,
        fs=real_array_signal.fs,
        duration=0.1,
    )

    # Force calculation
    asm.calculate()

    # Create mock mic signals (Time domain)
    # Shape: (Time, Mics)
    n_samples = 1000
    n_mics = real_array_signal.n_channels
    mic_signals = np.random.randn(n_samples, n_mics)

    encoded = asm.encode_amb(mic_signals)

    # encoded is SpatialSignal
    assert encoded.is_time
    assert encoded.is_sh

    L = (sh_order + 1) ** 2
    # SpatialSignal data is (Channels, Grid, Time); encoded Ambisonics has
    # Channels = L SH coefficients and a singleton grid axis.
    # Check shape
    assert encoded.data.shape[:2] == (1, L)
    encodedf = np.fft.fft(encoded.data, axis=-1)
    assert is_signal_frequency_sh_valid(encodedf, freq_axis=-1)
