"""
Tests for BSM (Beamformer-Steered Matching) and BSMEncoder.

Note on fixture design: SphericalArray steering matrices are physically rank-1 at
DC (f=0) because all Bn radial functions collapse to the n=0 term there, making
every column of V identical. The tikhonov solver in bsm.py does not guard against
this, so we use a synthetic full-rank complex steering matrix to exercise BSM's
algorithm across all frequencies including DC.
"""

import pytest
import numpy as np
from shroom.encoders.bsm import BSM, calculate_bsm_coefficients
from shroom.acoustics.processors import BSMEncoder
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.utils.dsp_utils import is_signal_frequency_symmetric


# ---------------------------------------------------------------------------
# Shared fixture: synthetic full-rank steering matrix + HRTF
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bsm_setup():
    """
    Deterministic BSM setup using a synthetic (full-rank everywhere) steering
    matrix instead of a physical SphericalArray, which is rank-1 at DC.
    """
    fs = 8000
    n_fft = 80     # 80 samples @ 8 kHz = 0.01 s
    Q = 50         # source grid points
    n_mics = 5

    source_grid = from_fibonacci_grid(Q)

    rng = np.random.default_rng(42)

    # Synthetic steering matrix: (M, Q, F) complex — full rank at every frequency
    V = rng.standard_normal((n_mics, Q, n_fft)) + 1j * rng.standard_normal((n_mics, Q, n_fft))
    array_mock = SpatialSignal(
        data=V, fs=fs, is_time=False, is_space=True, grid=source_grid
    )

    # Synthetic HRTF: (2, Q, F) complex
    H = rng.standard_normal((2, Q, n_fft)) + 1j * rng.standard_normal((2, Q, n_fft))
    hrtf_mock = SpatialSignal(
        data=H, fs=fs, is_time=False, is_space=True, grid=source_grid
    )

    bsm = BSM(array=array_mock, hrtf=hrtf_mock, use_magls=False, fs=fs)
    bsm_magls = BSM(
        array=array_mock,
        hrtf=hrtf_mock,
        use_magls=True,
        magls_cutoff_frequency=500.0,
        fs=fs,
    )

    return {
        "fs": fs,
        "n_fft": n_fft,
        "n_mics": n_mics,
        "Q": Q,
        "array": array_mock,
        "hrtf": hrtf_mock,
        "bsm": bsm,
        "bsm_magls": bsm_magls,
    }


# ---------------------------------------------------------------------------
# calculate_bsm_coefficients (standalone function)
# ---------------------------------------------------------------------------

def test_calculate_bsm_coefficients_shape(bsm_setup):
    """Output filters have the expected full-spectrum shape (F, M)."""
    s = bsm_setup
    cl, cr = calculate_bsm_coefficients(V=s["array"].data, h=s["hrtf"].data, fs=s["fs"])
    F, M = s["n_fft"], s["n_mics"]
    assert cl.shape == (F, M), f"cl shape {cl.shape} != ({F}, {M})"
    assert cr.shape == (F, M), f"cr shape {cr.shape} != ({F}, {M})"
    assert cl.dtype == np.complex128
    assert cr.dtype == np.complex128


def test_calculate_bsm_coefficients_dc_nyquist_real(bsm_setup):
    """DC (f=0) and Nyquist bins must be real-valued."""
    s = bsm_setup
    cl, cr = calculate_bsm_coefficients(V=s["array"].data, h=s["hrtf"].data, fs=s["fs"])
    F = s["n_fft"]
    np.testing.assert_allclose(cl[0].imag, 0, atol=1e-10, err_msg="cl DC not real")
    np.testing.assert_allclose(cr[0].imag, 0, atol=1e-10, err_msg="cr DC not real")
    if F % 2 == 0:
        np.testing.assert_allclose(cl[F // 2].imag, 0, atol=1e-10, err_msg="cl Nyquist not real")
        np.testing.assert_allclose(cr[F // 2].imag, 0, atol=1e-10, err_msg="cr Nyquist not real")


def test_calculate_bsm_coefficients_hermitian_symmetry(bsm_setup):
    """Full-spectrum filters must be Hermitian-symmetric (IFFT gives real output)."""
    s = bsm_setup
    cl, cr = calculate_bsm_coefficients(V=s["array"].data, h=s["hrtf"].data, fs=s["fs"])
    # is_signal_frequency_symmetric expects freq on the last axis
    assert is_signal_frequency_symmetric(cl.T, freq_axis=-1), "cl not Hermitian symmetric"
    assert is_signal_frequency_symmetric(cr.T, freq_axis=-1), "cr not Hermitian symmetric"


# ---------------------------------------------------------------------------
# BSM class
# ---------------------------------------------------------------------------

def test_bsm_class_coefficients_shape(bsm_setup):
    """BSM class produces filters with the expected shape."""
    s = bsm_setup
    cl, cr = s["bsm"].get_coefficients()
    F, M = s["n_fft"], s["n_mics"]
    assert cl.shape == (F, M)
    assert cr.shape == (F, M)


def test_bsm_class_matches_standalone_function(bsm_setup):
    """BSM class and standalone function produce identical results."""
    s = bsm_setup
    cl_class, cr_class = s["bsm"].get_coefficients()
    cl_func, cr_func = calculate_bsm_coefficients(
        V=s["array"].data, h=s["hrtf"].data, fs=s["fs"]
    )
    np.testing.assert_array_equal(cl_class, cl_func)
    np.testing.assert_array_equal(cr_class, cr_func)


def test_bsm_with_magls_shape(bsm_setup):
    """BSM with MagLS enabled still produces full-spectrum filters."""
    s = bsm_setup
    cl, cr = s["bsm_magls"].get_coefficients()
    assert cl.shape == (s["n_fft"], s["n_mics"])
    assert cr.shape == (s["n_fft"], s["n_mics"])


def test_bsm_with_magls_dc_nyquist_real(bsm_setup):
    """MagLS BSM: DC and Nyquist must still be real-valued."""
    s = bsm_setup
    cl, cr = s["bsm_magls"].get_coefficients()
    F = s["n_fft"]
    np.testing.assert_allclose(cl[0].imag, 0, atol=1e-10)
    np.testing.assert_allclose(cr[0].imag, 0, atol=1e-10)
    if F % 2 == 0:
        np.testing.assert_allclose(cl[F // 2].imag, 0, atol=1e-10)
        np.testing.assert_allclose(cr[F // 2].imag, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# BSM.process()
# ---------------------------------------------------------------------------

@pytest.fixture
def mic_signals(bsm_setup):
    """Deterministic mock microphone signals in time domain."""
    rng = np.random.default_rng(7)
    T = 200
    M = bsm_setup["n_mics"]
    data = rng.standard_normal((M, 1, T))
    return SpatialSignal(data=data, fs=bsm_setup["fs"], is_time=True, is_space=False)


def test_bsm_process_output_shape(bsm_setup, mic_signals):
    """BSM.process() returns a (2, 1, T) binaural SpatialSignal."""
    binaural = bsm_setup["bsm"].process(mic_signals)
    T = mic_signals.data.shape[2]
    assert binaural.data.shape == (2, 1, T), f"Expected (2, 1, {T}), got {binaural.data.shape}"


def test_bsm_process_output_domain(bsm_setup, mic_signals):
    """BSM.process() output is time-domain, not-space."""
    binaural = bsm_setup["bsm"].process(mic_signals)
    assert binaural.is_time
    assert not binaural.is_space


def test_bsm_process_requires_time_domain(bsm_setup, mic_signals):
    """BSM.process() raises ValueError when input is frequency-domain."""
    freq_mics = mic_signals.copy()
    freq_mics.toFreq()
    with pytest.raises(ValueError):
        bsm_setup["bsm"].process(freq_mics)


# ---------------------------------------------------------------------------
# BSMEncoder processor
# ---------------------------------------------------------------------------

def test_bsm_encoder_output_matches_bsm_process(bsm_setup, mic_signals):
    """BSMEncoder.process() and BSM.process() return identical arrays."""
    encoder = BSMEncoder(bsm_setup["bsm"])
    out_encoder = encoder.process(mic_signals)
    out_bsm = bsm_setup["bsm"].process(mic_signals)
    np.testing.assert_array_equal(out_encoder.data, out_bsm.data)


def test_bsm_encoder_fs_property(bsm_setup):
    """BSMEncoder.fs returns the array's sampling rate."""
    encoder = BSMEncoder(bsm_setup["bsm"])
    assert encoder.fs == bsm_setup["fs"]
