"""test_aria_asm_bsm.py
=====================
Integration tests for ASM and BSM encoders using the real ARIA array ATFs.

Validates:
- ARIA array loading, shape, finiteness
- Frequency-domain Hermitian symmetry
- SH-domain frequency validity
- Intermediate pipeline signals: space_hrtf, magls_result, aria_sh, cnm, and
  per-step processor outputs (ArrayDecoder, ASMEncoder, BinauralDecoder)
- ASM filter shape, DC/Nyquist constraints, and SH reconstruction accuracy
- BSM filter shape, DC/Nyquist constraints, and binaural reconstruction accuracy
- ProcessorChain full chain == step-by-step numerical equivalence
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "projects"))

from shroom.utils.sofa import load_sofa
from shroom.utils.file_utils import load_file
from shroom.paths import DEFAULT_HRTF_PATH
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.utils.dsp_utils import is_signal_frequency_symmetric, is_signal_frequency_sh_valid
from shroom.acoustics.processors import ProcessorChain, ArrayDecoder, ASMEncoder, BinauralDecoder
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom_dev.errors import asm_mse_error, bsm_mse_error

ARIA_PATH = os.path.join(REPO_ROOT, "data", "sofa_arrays", "aria_atfs_fixed.sofa")
SH_ORDER = 1
MAGLS_CUTOFF = 1200.0
SM_SH_ORDER = 20


@pytest.fixture(scope="module")
def aria_data():
    """Load ARIA ATFs once for all tests."""
    aria = load_sofa(ARIA_PATH)
    return aria


@pytest.fixture(scope="module")
def aria_freq(aria_data):
    """ARIA in frequency domain."""
    a = aria_data.copy()
    a.toFreq()
    return a


@pytest.fixture(scope="module")
def aria_sh(aria_data):
    """ARIA in SH domain (time)."""
    a = aria_data.copy()
    a.toSH(SM_SH_ORDER)
    return a


@pytest.fixture(scope="module")
def space_hrtf(aria_data):
    """HRTF projected onto the ARIA source grid."""
    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(aria_data.fs)
    n_fft = aria_data.data.shape[-1]
    hrtf.zero_pad(n_fft)
    hrtf.toFreq()
    hrtf.toSH(30)
    hrtf.toSpace(aria_data.grid)
    return hrtf


@pytest.fixture(scope="module")
def asm_encoder(aria_freq):
    """ASM encoder computed on ARIA (N=1)."""
    n_fft = aria_freq.data.shape[-1]
    duration = n_fft / aria_freq.fs
    return ASM(sh_order=SH_ORDER, array=aria_freq, fs=aria_freq.fs, duration=duration)


@pytest.fixture(scope="module")
def bsm_encoder(aria_freq, space_hrtf):
    """BSM+MagLS encoder computed on ARIA."""
    n_fft = aria_freq.data.shape[-1]
    duration = n_fft / aria_freq.fs
    return BSM(
        array=aria_freq,
        hrtf=space_hrtf,
        use_magls=True,
        magls_cutoff_frequency=MAGLS_CUTOFF,
        fs=aria_freq.fs,
        duration=duration,
    )


@pytest.fixture(scope="module")
def magls_result(space_hrtf):
    """MagLS HRTF for ASM rendering."""
    return magls_hrtf(hrtf=space_hrtf, sh_order=SH_ORDER, cutoff_over_freq=MAGLS_CUTOFF)


# ---------------------------------------------------------------------------
# ARIA array validation
# ---------------------------------------------------------------------------

def test_aria_shape(aria_data):
    """ARIA ATFs have expected 3-D shape (M, Q, T)."""
    assert aria_data.data.ndim == 3
    M, Q, T = aria_data.data.shape
    assert M > 0 and Q > 0 and T > 0


def test_aria_finite(aria_data):
    """ARIA ATFs contain no NaN/Inf."""
    assert np.all(np.isfinite(aria_data.data))


def test_aria_freq_hermitian_symmetry(aria_freq):
    """Frequency-domain ARIA satisfies Hermitian symmetry (real time signal)."""
    # Check a representative subset of channels to keep the test fast
    for m in range(aria_freq.data.shape[0]):
        assert is_signal_frequency_symmetric(aria_freq.data[m], freq_axis=-1), \
            f"Mic {m} failed Hermitian symmetry check."


def test_aria_sh_frequency_validity(aria_freq):
    """ARIA converted to SH domain satisfies the SH-frequency reality condition."""
    aria_sh_freq = aria_freq.copy()
    aria_sh_freq.toSH(SH_ORDER)
    # data shape: (1, nm, F) — channels on axis 0, sh on axis 1, freq on axis 2
    assert is_signal_frequency_sh_valid(aria_sh_freq.data, freq_axis=2, sh_axis=1)


# ---------------------------------------------------------------------------
# Intermediate signal validation
# ---------------------------------------------------------------------------

def test_space_hrtf_finite(space_hrtf):
    """HRTF projected onto ARIA grid contains no NaN/Inf."""
    assert np.all(np.isfinite(space_hrtf.data))


def test_space_hrtf_hermitian_symmetry(space_hrtf):
    """HRTF (freq+space) satisfies Hermitian symmetry per ear — real time-domain filters."""
    for ear in range(space_hrtf.data.shape[0]):
        assert is_signal_frequency_symmetric(space_hrtf.data[ear], freq_axis=-1), \
            f"space_hrtf ear {ear} failed Hermitian symmetry."


def test_magls_hrtf_finite(magls_result):
    """MagLS HRTF contains no NaN/Inf."""
    assert np.all(np.isfinite(magls_result.data))


def test_magls_hrtf_sh_freq_validity(magls_result):
    """MagLS HRTF (time+SH) satisfies the SH+freq reality condition after FFT."""
    m = magls_result.copy()
    m.toFreq()
    # shape: (2, nm, F) — ears on axis 0, SH on axis 1, freq on axis 2
    assert is_signal_frequency_sh_valid(m.data, freq_axis=2, sh_axis=1), \
        "MagLS HRTF failed SH+freq validity."


def test_aria_sh_finite(aria_sh):
    """ARIA in time+SH domain contains no NaN/Inf."""
    assert np.all(np.isfinite(aria_sh.data))


def test_aria_sh_freq_validity(aria_sh):
    """ARIA (time+SH, high order) satisfies SH+freq reality condition after FFT."""
    a = aria_sh.copy()
    a.toFreq()
    # shape: (M, nm, F)
    assert is_signal_frequency_sh_valid(a.data, freq_axis=2, sh_axis=1), \
        "ARIA (high-order SH) failed SH+freq validity."


def test_asm_cnm_sh_freq_validity(asm_encoder):
    """ASM cnm (freq+SH) satisfies the SH+freq reality condition."""
    # shape: (M, nm, F)
    assert is_signal_frequency_sh_valid(asm_encoder.cnm.data, freq_axis=2, sh_axis=1), \
        "ASM cnm failed SH+freq validity."


@pytest.fixture(scope="module")
def sh_impulse(aria_sh):
    """A first-order SH impulse (n=0,m=0) as SpatialSignal (1, nm, T), time+SH."""
    nm = (SM_SH_ORDER + 1) ** 2
    T = aria_sh.data.shape[-1]
    data = np.zeros((1, nm, T), dtype=np.float64)
    data[0, 0, 0] = 1.0
    return SpatialSignal(data=data, fs=aria_sh.fs, is_time=True, is_space=False)


def test_array_decoder_output_hermitian(aria_sh, sh_impulse):
    """ArrayDecoder output (mic signals) satisfies Hermitian symmetry — real time-domain."""
    decoder = ArrayDecoder(aria_sh, sh_order=1)
    mic_out = decoder.process(sh_impulse)
    mic_freq = mic_out.copy()
    mic_freq.toFreq()
    for m in range(mic_freq.data.shape[0]):
        assert is_signal_frequency_symmetric(mic_freq.data[m], freq_axis=-1), \
            f"ArrayDecoder mic {m} output failed Hermitian symmetry."


def test_asm_encoder_output_sh_freq_validity(asm_encoder, impulse_mic_signal):
    """ASMEncoder output (ambisonics) satisfies the SH+freq reality condition."""
    proc = ASMEncoder(asm_encoder)
    amb_out = proc.process(impulse_mic_signal)
    freq = amb_out.copy()
    freq.toFreq()
    # shape: (1, nm, F)
    assert is_signal_frequency_sh_valid(freq.data, freq_axis=2, sh_axis=1), \
        "ASMEncoder output failed SH+freq validity."


def test_binaural_decoder_output_hermitian(asm_encoder, magls_result, impulse_mic_signal):
    """BinauralDecoder output (binaural) satisfies Hermitian symmetry — real time-domain."""
    asm_proc = ASMEncoder(asm_encoder)
    binaural_proc = BinauralDecoder(magls_result, output_format="SpatialSignal")
    amb_out = asm_proc.process(impulse_mic_signal)
    binaural_out = binaural_proc.process(amb_out)
    binaural_freq = binaural_out.copy()
    binaural_freq.toFreq()
    for ear in range(binaural_freq.data.shape[0]):
        assert is_signal_frequency_symmetric(binaural_freq.data[ear], freq_axis=-1), \
            f"BinauralDecoder ear {ear} output failed Hermitian symmetry."


# ---------------------------------------------------------------------------
# ASM filter validation
# ---------------------------------------------------------------------------

def test_asm_shape(asm_encoder, aria_freq):
    """ASM coefficients have shape (M, nm, F)."""
    cnm = asm_encoder.cnm
    M = aria_freq.data.shape[0]
    nm = (SH_ORDER + 1) ** 2
    F = aria_freq.data.shape[-1]
    assert cnm.data.shape == (M, nm, F), f"Expected ({M}, {nm}, {F}), got {cnm.data.shape}"


def test_asm_finite(asm_encoder):
    """ASM coefficients contain no NaN/Inf."""
    assert np.all(np.isfinite(asm_encoder.cnm.data))


def test_asm_dc_constraint(asm_encoder):
    """ASM DC bin: higher-order channels are zero, omnidirectional is real."""
    cnm = asm_encoder.cnm.data  # (M, nm, F)
    # Higher-order SH channels (nm index > 0) at DC (f=0) must be zero
    assert np.allclose(cnm[:, 1:, 0], 0.0), "ASM DC: higher-order channels are not zero."
    # Omnidirectional (nm=0) at DC must be real
    assert np.allclose(cnm[:, 0, 0].imag, 0.0), "ASM DC: (0,0) channel is not real."


def test_asm_nyquist_constraint(asm_encoder):
    """ASM Nyquist bin: higher-order channels are zero, omnidirectional is real."""
    cnm = asm_encoder.cnm.data  # (M, nm, F)
    F = cnm.shape[-1]
    if F % 2 == 0:
        nyq = F // 2
        assert np.allclose(cnm[:, 1:, nyq], 0.0), "ASM Nyquist: higher-order channels are not zero."
        assert np.allclose(cnm[:, 0, nyq].imag, 0.0), "ASM Nyquist: (0,0) channel is not real."


def test_asm_reconstruction_accuracy(asm_encoder, aria_freq):
    """ASM SH reconstruction error is bounded at mid-frequencies."""
    cnm = asm_encoder.cnm.data          # (M, nm, F)
    sm = aria_freq.data                 # (M, Q, F)
    Y = aria_freq.grid.Y(N_sp=SH_ORDER)  # (Q, nm)
    freqs = np.fft.fftfreq(sm.shape[-1], 1.0 / aria_freq.fs)

    error = asm_mse_error(cnm, sm, Y, freqs)  # (nm, F_pos)

    # Omnidirectional channel (n=0,m=0): median relative error should be modest
    # (ARIA is a well-conditioned array so this should hold well)
    pos_freqs = np.fft.rfftfreq(sm.shape[-1], 1.0 / aria_freq.fs)
    mid_mask = (pos_freqs >= 200) & (pos_freqs <= 4000)
    median_err_omni = np.median(error[0, mid_mask])
    assert median_err_omni < 1.0, (
        f"ASM omnidirectional reconstruction error too large: {median_err_omni:.4f}"
    )


# ---------------------------------------------------------------------------
# BSM filter validation
# ---------------------------------------------------------------------------

def test_bsm_coefficients_shape(bsm_encoder, aria_freq):
    """BSM coefficients have shape (F, M) per ear."""
    cl, cr = bsm_encoder.get_coefficients()
    F = aria_freq.data.shape[-1]
    M = aria_freq.data.shape[0]
    assert cl.shape == (F, M), f"BSM cl expected ({F}, {M}), got {cl.shape}"
    assert cr.shape == (F, M), f"BSM cr expected ({F}, {M}), got {cr.shape}"


def test_bsm_coefficients_finite(bsm_encoder):
    """BSM coefficients contain no NaN/Inf."""
    cl, cr = bsm_encoder.get_coefficients()
    assert np.all(np.isfinite(cl))
    assert np.all(np.isfinite(cr))


def test_bsm_dc_real(bsm_encoder):
    """BSM DC bin (f=0) must be real for both ears."""
    cl, cr = bsm_encoder.get_coefficients()
    assert np.allclose(cl[0].imag, 0.0, atol=1e-8), "BSM cl DC is not real."
    assert np.allclose(cr[0].imag, 0.0, atol=1e-8), "BSM cr DC is not real."


def test_bsm_nyquist_real(bsm_encoder):
    """BSM Nyquist bin must be real for both ears (even-length FFT)."""
    cl, cr = bsm_encoder.get_coefficients()
    F = cl.shape[0]
    if F % 2 == 0:
        nyq = F // 2
        assert np.allclose(cl[nyq].imag, 0.0, atol=1e-8), "BSM cl Nyquist is not real."
        assert np.allclose(cr[nyq].imag, 0.0, atol=1e-8), "BSM cr Nyquist is not real."


def test_bsm_hermitian_symmetry(bsm_encoder):
    """BSM coefficients satisfy Hermitian symmetry (real time-domain filters)."""
    cl, cr = bsm_encoder.get_coefficients()
    assert is_signal_frequency_symmetric(cl, freq_axis=0), "BSM cl failed Hermitian symmetry."
    assert is_signal_frequency_symmetric(cr, freq_axis=0), "BSM cr failed Hermitian symmetry."


def test_bsm_reconstruction_accuracy(bsm_encoder, aria_freq, space_hrtf):
    """BSM binaural reconstruction error is bounded at mid-frequencies."""
    cl, cr = bsm_encoder.get_coefficients()
    freqs = np.fft.fftfreq(aria_freq.data.shape[-1], 1.0 / aria_freq.fs)
    pos_freqs = np.fft.rfftfreq(aria_freq.data.shape[-1], 1.0 / aria_freq.fs)

    mse_l, mse_r = bsm_mse_error(cl, cr, aria_freq, space_hrtf, freqs)

    # BSM minimises MSE directly below the MagLS cutoff — check performance there.
    low_mask = (pos_freqs >= 200) & (pos_freqs <= MAGLS_CUTOFF)
    assert np.median(mse_l[low_mask]) < 1.0, \
        f"BSM left-ear reconstruction error too large below MagLS cutoff: {np.median(mse_l[low_mask]):.4f}"
    assert np.median(mse_r[low_mask]) < 1.0, \
        f"BSM right-ear reconstruction error too large below MagLS cutoff: {np.median(mse_r[low_mask]):.4f}"


# ---------------------------------------------------------------------------
# ProcessorChain equivalence
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def impulse_mic_signal(aria_sh):
    """A single-mic impulse as SpatialSignal (M, 1, T), for chain equivalence test."""
    M = aria_sh.data.shape[0]
    T = aria_sh.data.shape[-1]
    impulse = np.zeros((M, 1, T), dtype=np.float64)
    impulse[0, 0, 0] = 1.0
    return SpatialSignal(
        data=impulse,
        fs=aria_sh.fs,
        is_time=True,
        is_space=False,
    )


def test_processor_chain_vs_sequential(
    aria_sh, asm_encoder, magls_result, impulse_mic_signal
):
    """Full ProcessorChain output matches sequential processor calls."""
    # Build processors
    asm_proc = ASMEncoder(asm_encoder)
    binaural_proc = BinauralDecoder(magls_result, output_format="SpatialSignal")
    chain = ProcessorChain([asm_proc, binaural_proc])

    # Step-by-step
    step1 = asm_proc.process(impulse_mic_signal)
    step2 = binaural_proc.process(step1)

    # Full chain
    chain_out = chain.process(impulse_mic_signal)

    seq_data = step2.data if isinstance(step2, SpatialSignal) else step2
    chain_data = chain_out.data if isinstance(chain_out, SpatialSignal) else chain_out

    np.testing.assert_allclose(
        chain_data, seq_data, atol=1e-5,
        err_msg="ProcessorChain output does not match step-by-step output."
    )


def test_binaural_output_real_finite(aria_sh, asm_encoder, magls_result, impulse_mic_signal):
    """Binaural output is finite and real-valued."""
    asm_proc = ASMEncoder(asm_encoder)
    binaural_proc = BinauralDecoder(magls_result, output_format="SpatialSignal")
    chain = ProcessorChain([asm_proc, binaural_proc])
    out = chain.process(impulse_mic_signal)
    data = out.data if isinstance(out, SpatialSignal) else out
    assert np.all(np.isfinite(data)), "Binaural output contains NaN or Inf."
    assert np.allclose(data.imag, 0, atol=1e-6), "Binaural output has non-negligible imaginary part."
