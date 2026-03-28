"""
Comprehensive tests for SpatialSignal.

Covers every public method across many fs values, signal durations, and
edge cases including odd/prime lengths, non-integer resample ratios, very
short signals, and mathematical identities (Parseval, Hermitian symmetry,
SH round-trip, rotation consistency).
"""
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_equator_grid(n_points: int) -> sphereicalGrid:
    az = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    co = np.full(n_points, np.pi / 2)
    return sphereicalGrid(az, co)


def make_time_space(n_ch, n_pts, n_samp, fs, seed=0) -> SpatialSignal:
    rng = np.random.default_rng(seed)
    grid = make_equator_grid(n_pts)
    data = rng.standard_normal((n_ch, n_pts, n_samp))
    return SpatialSignal(data, fs, is_time=True, is_space=True, grid=grid)


def make_time_sh(n_ch, sh_order, n_samp, fs, seed=0) -> SpatialSignal:
    rng = np.random.default_rng(seed)
    n_nm = (sh_order + 1) ** 2
    data = rng.standard_normal((n_ch, n_nm, n_samp))
    return SpatialSignal(data, fs, is_time=True, is_space=False)


def make_sh_subspace_signal(n_ch, sh_order, n_samp, fs, grid, seed=0) -> SpatialSignal:
    """Space-domain signal that lies exactly in the SH subspace of given order."""
    rng = np.random.default_rng(seed)
    n_nm = (sh_order + 1) ** 2
    Y = grid.Y(sh_order)                             # (Q, nm)
    sh_data = rng.standard_normal((n_ch, n_nm, n_samp))
    space_data = np.einsum("qn,cnt->cqt", Y, sh_data).real  # (n_ch, Q, n_samp)
    return SpatialSignal(space_data, fs, is_time=True, is_space=True, grid=grid)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter sets
# ─────────────────────────────────────────────────────────────────────────────

FS_VALUES = [8000, 16000, 22050, 44100, 48000, 96000]

N_SAMPLES = pytest.mark.parametrize("n_samp", [
    pytest.param(16,   id="16"),
    pytest.param(32,   id="32"),
    pytest.param(64,   id="64"),
    pytest.param(100,  id="100_even"),
    pytest.param(127,  id="127_prime"),
    pytest.param(128,  id="128_pow2"),
    pytest.param(257,  id="257_odd"),
    pytest.param(512,  id="512_pow2"),
    pytest.param(1000, id="1000"),
])

RESAMPLE_PAIRS = pytest.mark.parametrize("fs_pair", [
    pytest.param((48000, 24000), id="48k->24k"),
    pytest.param((48000, 16000), id="48k->16k"),
    pytest.param((48000, 12000), id="48k->12k"),
    pytest.param((16000, 48000), id="16k->48k"),
    pytest.param((22050, 44100), id="22k->44k"),
    pytest.param((44100, 16000), id="44k->16k_nonint"),
    pytest.param((16000, 22050), id="16k->22k_nonint"),
    pytest.param((8000,  48000), id="8k->48k_large"),
    pytest.param((96000, 16000), id="96k->16k_large"),
])

SH_ORDERS = pytest.mark.parametrize("sh_order", [1, 2, 3, 4])


# ═════════════════════════════════════════════════════════════════════════════
# Construction & validation
# ═════════════════════════════════════════════════════════════════════════════

class TestConstruction:

    @pytest.mark.parametrize("fs", FS_VALUES)
    @pytest.mark.parametrize("n_samp", [16, 128, 1000])
    def test_time_space_domain_flags(self, fs, n_samp):
        sig = make_time_space(2, 8, n_samp, fs)
        assert sig.is_time and not sig.is_freq
        assert sig.is_space and not sig.is_sh
        assert sig.fs == fs
        assert sig.n_samples == n_samp
        assert sig.data.shape == (2, 8, n_samp)

    @pytest.mark.parametrize("fs", FS_VALUES)
    def test_freq_sh_domain_flags(self, fs):
        n_nm = (2 + 1) ** 2
        data = np.random.randn(2, n_nm, 64) + 1j * np.random.randn(2, n_nm, 64)
        sig = SpatialSignal(data, fs, is_time=False, is_space=False)
        assert sig.is_freq and not sig.is_time
        assert sig.is_sh and not sig.is_space
        assert sig.sh_order == 2

    def test_invalid_space_without_grid(self):
        with pytest.raises(AssertionError):
            SpatialSignal(np.zeros((1, 8, 64)), 48000, is_time=True, is_space=True, grid=None)

    def test_invalid_sh_with_grid(self):
        grid = make_equator_grid(8)
        with pytest.raises(AssertionError):
            SpatialSignal(np.zeros((1, 8, 64)), 48000, is_time=True, is_space=False, grid=grid)

    def test_invalid_data_not_3d(self):
        grid = make_equator_grid(8)
        with pytest.raises(AssertionError):
            SpatialSignal(np.zeros((8, 64)), 48000, is_time=True, is_space=True, grid=grid)

    def test_invalid_grid_shape_mismatch(self):
        grid = make_equator_grid(8)
        with pytest.raises(AssertionError):
            SpatialSignal(np.zeros((1, 10, 64)), 48000, is_time=True, is_space=True, grid=grid)

    def test_invalid_negative_fs(self):
        grid = make_equator_grid(4)
        with pytest.raises(AssertionError):
            SpatialSignal(np.zeros((1, 4, 64)), -1, is_time=True, is_space=True, grid=grid)


# ═════════════════════════════════════════════════════════════════════════════
# copy()
# ═════════════════════════════════════════════════════════════════════════════

class TestCopy:

    def test_data_is_independent(self):
        sig = make_time_space(1, 8, 128, 48000)
        cp = sig.copy()
        cp.data[0, 0, 0] = 9999.0
        assert sig.data[0, 0, 0] != 9999.0

    @pytest.mark.parametrize("fs", FS_VALUES)
    def test_preserves_fs(self, fs):
        assert make_time_space(1, 4, 64, fs).copy().fs == fs

    def test_preserves_domain_flags_after_conversion(self):
        sig = make_time_space(2, 8, 64, 48000)
        sig.toFreq()
        cp = sig.copy()
        assert cp.is_freq and cp.is_space

    def test_grid_is_deeply_copied(self):
        sig = make_time_space(1, 8, 64, 48000)
        cp = sig.copy()
        cp.grid.az[0] = 999.0
        assert sig.grid.az[0] != 999.0

    def test_grid_is_none_for_sh_domain(self):
        cp = make_time_sh(2, 2, 64, 48000).copy()
        assert cp.grid is None

    def test_history_is_independent(self):
        sig = make_time_space(1, 4, 64, 48000)
        cp = sig.copy()
        cp._history.append({"test": True})
        assert len(sig._history) != len(cp._history)


# ═════════════════════════════════════════════════════════════════════════════
# zero_pad()
# ═════════════════════════════════════════════════════════════════════════════

class TestZeroPad:

    @pytest.mark.parametrize("n_orig,n_target", [
        (64,  128),
        (100, 200),
        (127, 256),
        (128, 512),
        (13,  100),
        (1,   1000),
    ])
    def test_final_length(self, n_orig, n_target):
        sig = make_time_space(1, 4, n_orig, 48000)
        sig.zero_pad(n_target)
        assert sig.data.shape[-1] == n_target

    @pytest.mark.parametrize("n_orig", [16, 64, 128, 256])
    def test_original_samples_unchanged(self, n_orig):
        sig = make_time_space(2, 6, n_orig, 48000, seed=7)
        original = sig.data.copy()
        sig.zero_pad(n_orig * 2)
        np.testing.assert_array_equal(sig.data[..., :n_orig], original)

    @pytest.mark.parametrize("n_orig", [16, 64, 128])
    def test_appended_samples_are_zero(self, n_orig):
        sig = make_time_space(1, 4, n_orig, 48000)
        sig.zero_pad(n_orig + 50)
        np.testing.assert_array_equal(sig.data[..., n_orig:], 0.0)

    @pytest.mark.parametrize("n_samp", [16, 64, 128, 513])
    def test_noop_at_exact_same_length(self, n_samp):
        sig = make_time_space(1, 4, n_samp, 48000)
        original = sig.data.copy()
        sig.zero_pad(n_samp)
        np.testing.assert_array_equal(sig.data, original)

    def test_raises_when_target_shorter(self):
        sig = make_time_space(1, 4, 128, 48000)
        with pytest.raises(ValueError):
            sig.zero_pad(64)

    def test_warns_and_converts_from_freq_domain(self):
        sig = make_time_space(1, 4, 64, 48000)
        sig.toFreq()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.zero_pad(128)
        assert any(w), "Expected a warning when zero-padding in freq domain"
        assert sig.is_time
        assert sig.data.shape[-1] == 128

    @pytest.mark.parametrize("fs", FS_VALUES)
    def test_works_for_all_fs(self, fs):
        sig = make_time_space(1, 4, 64, fs)
        sig.zero_pad(128)
        assert sig.data.shape[-1] == 128
        assert sig.fs == fs


# ═════════════════════════════════════════════════════════════════════════════
# resample()
# ═════════════════════════════════════════════════════════════════════════════

class TestResample:

    @RESAMPLE_PAIRS
    @pytest.mark.parametrize("n_samp", [64, 128, 256, 1000])
    def test_fs_is_updated(self, fs_pair, n_samp):
        fs_in, fs_out = fs_pair
        sig = make_time_space(1, 4, n_samp, fs_in)
        sig.resample(fs_out)
        assert sig.fs == fs_out

    @RESAMPLE_PAIRS
    @pytest.mark.parametrize("n_samp", [64, 128, 256])
    def test_expected_sample_count(self, fs_pair, n_samp):
        fs_in, fs_out = fs_pair
        sig = make_time_space(1, 4, n_samp, fs_in)
        sig.resample(fs_out)
        expected = int(n_samp * fs_out / fs_in)
        assert sig.data.shape[-1] == expected

    @pytest.mark.parametrize("fs", FS_VALUES)
    def test_noop_same_fs(self, fs):
        sig = make_time_space(1, 4, 128, fs)
        original = sig.data.copy()
        sig.resample(fs)
        np.testing.assert_array_equal(sig.data, original)

    @pytest.mark.parametrize("fs_pair", [
        (48000, 24000), (16000, 48000), (22050, 44100),
    ])
    def test_rms_approximately_preserved(self, fs_pair):
        """RMS should be preserved for a bandlimited sine after resampling."""
        fs_in, fs_out = fs_pair
        n = 512
        freq = min(fs_in, fs_out) / 8   # well within both Nyquist limits
        t = np.arange(n) / fs_in
        tone = np.sin(2 * np.pi * freq * t)
        grid = make_equator_grid(4)
        data = np.broadcast_to(tone, (1, 4, n)).copy()
        sig = SpatialSignal(data, fs_in, is_time=True, is_space=True, grid=grid)

        rms_before = np.sqrt(np.mean(sig.data ** 2))
        sig.resample(fs_out)
        rms_after = np.sqrt(np.mean(sig.data.real ** 2))
        np.testing.assert_allclose(rms_after, rms_before, rtol=0.05)

    @pytest.mark.parametrize("fs_pair", [
        (48000, 24000), (16000, 48000), (44100, 16000),
    ])
    def test_dominant_frequency_preserved(self, fs_pair):
        """Peak frequency bin should stay the same after resampling."""
        fs_in, fs_out = fs_pair
        n_in = 1024
        freq = min(fs_in, fs_out) / 4   # well within both Nyquist limits
        t = np.arange(n_in) / fs_in
        tone = np.sin(2 * np.pi * freq * t)
        grid = make_equator_grid(4)
        data = np.broadcast_to(tone, (1, 4, n_in)).copy()
        sig = SpatialSignal(data, fs_in, is_time=True, is_space=True, grid=grid)
        sig.resample(fs_out)

        n_out = sig.data.shape[-1]
        spectrum = np.abs(np.fft.rfft(sig.data[0, 0]))
        peak_freq = np.argmax(spectrum) * fs_out / n_out
        bin_width = fs_out / n_out
        assert abs(peak_freq - freq) <= 2 * bin_width

    def test_warns_and_converts_from_freq_domain(self):
        sig = make_time_space(1, 4, 128, 48000)
        sig.toFreq()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.resample(24000)
        assert any(w), "Expected a warning when resampling in freq domain"
        assert sig.is_time

    def test_result_is_always_time_domain(self):
        sig = make_time_space(1, 4, 128, 48000)
        sig.resample(16000)
        assert sig.is_time

    @pytest.mark.parametrize("n_samp", [16, 33, 64, 100, 127, 128])
    def test_short_and_edge_length_signals(self, n_samp):
        """Resampling should not crash for edge-case lengths."""
        sig = make_time_space(1, 4, n_samp, 48000)
        sig.resample(16000)
        assert sig.fs == 16000
        assert sig.data.shape[-1] == int(n_samp * 16000 / 48000)


# ═════════════════════════════════════════════════════════════════════════════
# toFreq() / toTime()
# ═════════════════════════════════════════════════════════════════════════════

class TestFreqTimeDomain:

    @N_SAMPLES
    @pytest.mark.parametrize("fs", [8000, 16000, 48000, 96000])
    def test_roundtrip_recovers_original(self, n_samp, fs):
        sig = make_time_space(2, 6, n_samp, fs, seed=1)
        original = sig.data.copy()
        sig.toFreq()
        assert sig.is_freq
        sig.toTime()
        assert sig.is_time
        np.testing.assert_allclose(sig.data.real, original, atol=1e-10)
        np.testing.assert_allclose(sig.data.imag, 0.0, atol=1e-10)

    @N_SAMPLES
    def test_tofreq_does_not_change_shape(self, n_samp):
        sig = make_time_space(2, 6, n_samp, 48000)
        shape_before = sig.data.shape
        sig.toFreq()
        assert sig.data.shape == shape_before

    @N_SAMPLES
    def test_hermitian_symmetry(self, n_samp):
        """X[k] == X[N-k]* for interior bins of real-signal FFT."""
        sig = make_time_space(1, 4, n_samp, 48000, seed=3)
        sig.toFreq()
        X = sig.data
        F = X.shape[-1]
        k = np.arange(1, F // 2)
        np.testing.assert_allclose(X[..., k], X[..., F - k].conj(), atol=1e-10)

    @N_SAMPLES
    def test_dc_bin_is_real(self, n_samp):
        sig = make_time_space(1, 4, n_samp, 48000)
        sig.toFreq()
        np.testing.assert_allclose(sig.data[..., 0].imag, 0.0, atol=1e-10)

    @pytest.mark.parametrize("n_samp", [16, 32, 64, 128, 256, 512, 1000])  # even only
    def test_nyquist_bin_is_real_for_even_n(self, n_samp):
        sig = make_time_space(1, 4, n_samp, 48000)
        sig.toFreq()
        np.testing.assert_allclose(sig.data[..., n_samp // 2].imag, 0.0, atol=1e-10)

    @pytest.mark.parametrize("n_samp", [15, 31, 63, 127, 257])  # odd only
    def test_dc_still_real_for_odd_n(self, n_samp):
        sig = make_time_space(1, 4, n_samp, 48000)
        sig.toFreq()
        np.testing.assert_allclose(sig.data[..., 0].imag, 0.0, atol=1e-10)

    @N_SAMPLES
    def test_parseval_theorem(self, n_samp):
        """sum|x[n]|^2 == sum|X[k]|^2 / N."""
        sig = make_time_space(1, 4, n_samp, 48000, seed=5)
        energy_time = np.sum(np.abs(sig.data) ** 2)
        sig.toFreq()
        energy_freq = np.sum(np.abs(sig.data) ** 2) / n_samp
        np.testing.assert_allclose(energy_freq, energy_time, rtol=1e-8)

    def test_tofreq_warns_if_already_freq(self):
        sig = make_time_space(1, 4, 128, 48000)
        sig.toFreq()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.toFreq()
        assert len(w) == 1

    def test_totime_warns_if_already_time(self):
        sig = make_time_space(1, 4, 128, 48000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.toTime()
        assert len(w) == 1

    def test_data_is_complex_after_tofreq(self):
        sig = make_time_space(1, 4, 128, 48000)
        sig.toFreq()
        assert np.iscomplexobj(sig.data)

    @pytest.mark.parametrize("fs", FS_VALUES)
    def test_roundtrip_all_fs(self, fs):
        sig = make_time_space(1, 4, 128, fs, seed=99)
        original = sig.data.copy()
        sig.toFreq()
        sig.toTime()
        np.testing.assert_allclose(sig.data.real, original, atol=1e-10)

    def test_tofreq_and_tosh_commute(self):
        """toFreq and toSH operate on different axes — order must not matter."""
        grid = from_fibonacci_grid(50)
        sig = make_sh_subspace_signal(2, 2, 64, 48000, grid)

        path_a = sig.copy()
        path_a.toSH(2)
        path_a.toFreq()

        path_b = sig.copy()
        path_b.toFreq()
        path_b.toSH(2)

        np.testing.assert_allclose(path_a.data, path_b.data, atol=1e-10)


# ═════════════════════════════════════════════════════════════════════════════
# toSH() / toSpace()
# ═════════════════════════════════════════════════════════════════════════════

class TestSHSpaceDomain:

    @SH_ORDERS
    def test_tosh_output_shape(self, sh_order):
        grid = from_fibonacci_grid((sh_order + 1) ** 2 * 4)
        data = np.random.randn(2, grid.n_points, 64)
        sig = SpatialSignal(data, 48000, is_time=True, is_space=True, grid=grid)
        sig.toSH(sh_order)
        assert sig.data.shape == (2, (sh_order + 1) ** 2, 64)

    @SH_ORDERS
    def test_tosh_domain_flags(self, sh_order):
        grid = from_fibonacci_grid((sh_order + 1) ** 2 * 4)
        data = np.random.randn(2, grid.n_points, 64)
        sig = SpatialSignal(data, 48000, is_time=True, is_space=True, grid=grid)
        sig.toSH(sh_order)
        assert sig.is_sh and not sig.is_space
        assert sig.grid is None

    @pytest.mark.parametrize("sh_order", [1, 2, 3])
    def test_tosh_tospace_roundtrip_on_subspace_signal(self, sh_order):
        """Signal in SH subspace must survive toSH → toSpace within regularisation error."""
        grid = from_fibonacci_grid(50)
        sig = make_sh_subspace_signal(2, sh_order, 64, 48000, grid)
        original = sig.data.copy()
        sig.toSH(sh_order)
        sig.toSpace(grid)
        # regularized_pinv uses lam = 0.01*sigma_max, so ~1e-3 relative error is expected
        np.testing.assert_allclose(sig.data, original, atol=1e-2)

    @pytest.mark.parametrize("sh_order", [1, 2, 3])
    def test_tospace_tosh_roundtrip(self, sh_order):
        """SH coefficients survive toSpace → toSH on the same well-sampled grid."""
        grid = from_fibonacci_grid(50)
        n_nm = (sh_order + 1) ** 2
        sh_data = np.random.randn(2, n_nm, 64)
        sig = SpatialSignal(sh_data, 48000, is_time=True, is_space=False)
        sig.toSpace(grid)
        sig.toSH(sh_order)
        np.testing.assert_allclose(sig.data, sh_data, atol=1e-7)

    def test_tosh_warns_if_already_sh(self):
        sig = make_time_sh(1, 2, 64, 48000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.toSH(2)
        assert len(w) == 1

    def test_toSpace_warns_if_already_space(self):
        sig = make_time_space(1, 8, 64, 48000)
        grid = make_equator_grid(8)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sig.toSpace(grid)
        assert len(w) == 1

    @pytest.mark.parametrize("sh_order", [1, 2])
    def test_tosh_in_freq_domain_does_not_change_freq_flag(self, sh_order):
        """toSH only acts on the spatial axis — freq flag must survive."""
        grid = from_fibonacci_grid(50)
        sig = make_sh_subspace_signal(2, sh_order, 64, 48000, grid)
        sig.toFreq()
        sig.toSH(sh_order)
        assert sig.is_sh and sig.is_freq

    @pytest.mark.parametrize("fs", [8000, 16000, 48000])
    @pytest.mark.parametrize("n_samp", [64, 128, 256])
    def test_roundtrip_many_fs_and_durations(self, fs, n_samp):
        sh_order = 2
        grid = from_fibonacci_grid(50)
        sig = make_sh_subspace_signal(1, sh_order, n_samp, fs, grid)
        original = sig.data.copy()
        sig.toSH(sh_order)
        sig.toSpace(grid)
        np.testing.assert_allclose(sig.data, original, atol=1e-6)

    @SH_ORDERS
    def test_sh_order_property(self, sh_order):
        assert make_time_sh(1, sh_order, 64, 48000).sh_order == sh_order

    def test_sh_order_is_none_in_space_domain(self):
        assert make_time_space(1, 8, 64, 48000).sh_order is None


# ═════════════════════════════════════════════════════════════════════════════
# convolve_sh()
# ═════════════════════════════════════════════════════════════════════════════

# class TestConvolveSH:
#
#     def test_identity_impulse_at_t0(self):
#         """Convolving with a unit impulse at t=0 on channel (0,0) yields that channel."""
#         fs, n_sh, T = 48000, 4, 50
#         data = np.random.randn(2, n_sh, T)
#         sig = SpatialSignal(data, fs, is_time=True, is_space=False)
#
#         filt_data = np.zeros((1, n_sh, T))
#         filt_data[0, 0, 0] = 1.0
#         filt = SpatialSignal(filt_data, fs, is_time=True, is_space=False)
#
#         out = sig.convolve_sh(filt)
#         # out[ch, 0, :T] is sum over SH channels — only ch 0 of sig contributes
#         np.testing.assert_allclose(out[:, 0, :T], data[:, 0, :], atol=1e-10)
#
#     def test_delay_shifts_output(self):
#         """Convolving an impulse signal with a delayed impulse shifts the peak."""
#         fs, n_sh, T, delay = 48000, 4, 50, 5
#         sig_data = np.zeros((1, n_sh, T))
#         sig_data[0, 0, 0] = 1.0
#         sig = SpatialSignal(sig_data, fs, is_time=True, is_space=False)
#
#         delayed_data = np.zeros((1, n_sh, T))
#         delayed_data[0, 0, delay] = 1.0
#         delayed = SpatialSignal(delayed_data, fs, is_time=True, is_space=False)
#
#         out = sig.convolve_sh(delayed)
#         assert np.argmax(np.abs(out[0, 0])) == delay
#
#     @pytest.mark.parametrize("T1,T2", [(50, 30), (100, 100), (10, 200), (1, 50)])
#     def test_output_length(self, T1, T2):
#         fs, n_sh = 48000, 4
#         s1 = SpatialSignal(np.random.randn(2, n_sh, T1), fs, is_time=True, is_space=False)
#         s2 = SpatialSignal(np.random.randn(3, n_sh, T2), fs, is_time=True, is_space=False)
#         out = s1.convolve_sh(s2)
#         assert out.shape == (2, 3, T1 + T2 - 1)
#
#     @pytest.mark.parametrize("order1,order2,expected", [
#         (3, 1, 1), (2, 2, 2), (4, 2, 2), (1, 3, 1),
#     ])
#     def test_sh_order_truncated_to_min(self, order1, order2, expected):
#         fs, T = 48000, 30
#         s1 = make_time_sh(1, order1, T, fs, seed=1)
#         s2 = make_time_sh(1, order2, T, fs, seed=2)
#         out = s1.convolve_sh(s2)
#         # Only `expected`-order channels participated in the sum
#         assert out.shape[2] == 2 * T - 1
#
#     def test_raises_for_space_domain_input(self):
#         sig = make_time_space(1, 8, 64, 48000)
#         filt = make_time_sh(1, 2, 64, 48000)
#         with pytest.raises(ValueError):
#             sig.convolve_sh(filt)
#
#     def test_works_when_signals_are_in_freq_domain(self):
#         """convolve_sh converts to time internally — freq input must not crash."""
#         fs, n_sh, T = 48000, 4, 50
#         sig_data = np.zeros((1, n_sh, T))
#         sig_data[0, 0, 0] = 1.0
#         s1 = SpatialSignal(sig_data.copy(), fs, is_time=False, is_space=False)
#         s2 = SpatialSignal(np.random.randn(2, n_sh, T), fs, is_time=False, is_space=False)
#         out = s1.convolve_sh(s2)
#         assert out.shape == (1, 2, 2 * T - 1)
#
#
# # ═════════════════════════════════════════════════════════════════════════════
# # rotate_space_domain()
# # ═════════════════════════════════════════════════════════════════════════════
#
# class TestRotateSpaceDomain:
#
#     def test_identity_rotation_leaves_data_unchanged(self):
#         sig = make_time_space(1, 8, 64, 48000, seed=10)
#         original = sig.data.copy()
#         sig.rotate_space_domain(Rotation.from_euler("z", 0, degrees=True))
#         np.testing.assert_allclose(sig.data, original, atol=1e-12)
#
#     def test_identity_rotation_leaves_grid_unchanged(self):
#         sig = make_time_space(1, 8, 64, 48000)
#         original_az = sig.grid.az.copy()
#         sig.rotate_space_domain(Rotation.from_euler("z", 0, degrees=True))
#         np.testing.assert_allclose(sig.grid.az, original_az, atol=1e-10)
#
#     @pytest.mark.parametrize("angle_deg", [30, 45, 90, 120, 180, 270, 360])
#     def test_z_rotation_shifts_all_azimuths(self, angle_deg):
#         az = np.linspace(0, 2 * np.pi, 8, endpoint=False)
#         grid = sphereicalGrid(az, np.full(8, np.pi / 2))
#         sig = SpatialSignal(np.random.randn(1, 8, 32), 48000,
#                             is_time=True, is_space=True, grid=grid)
#         sig.rotate_space_domain(Rotation.from_euler("z", angle_deg, degrees=True))
#         expected = np.mod(az + np.deg2rad(angle_deg), 2 * np.pi)
#         np.testing.assert_allclose(
#             np.exp(1j * sig.grid.az),
#             np.exp(1j * expected),
#             atol=1e-10,
#         )
#
#     def test_rotation_does_not_change_data_values(self):
#         """Rotating the grid must not alter the data array."""
#         sig = make_time_space(1, 8, 64, 48000, seed=55)
#         original_data = sig.data.copy()
#         sig.rotate_space_domain(Rotation.from_euler("z", 90, degrees=True))
#         np.testing.assert_array_equal(sig.data, original_data)
#
#     def test_forward_then_inverse_recovers_grid(self):
#         sig = make_time_space(1, 8, 64, 48000)
#         orig_az = sig.grid.az.copy()
#         orig_co = sig.grid.co.copy()
#         rot = Rotation.from_euler("zyx", [45, 20, 10], degrees=True)
#         sig.rotate_space_domain(rot)
#         sig.rotate_space_domain(rot.inv())
#         np.testing.assert_allclose(sig.grid.az, orig_az, atol=1e-10)
#         np.testing.assert_allclose(sig.grid.co, orig_co, atol=1e-10)
#
#     def test_colatitude_stays_in_valid_range(self):
#         sig = make_time_space(1, 16, 64, 48000)
#         for angle in [30, 60, 90, 120, 180]:
#             sig.rotate_space_domain(Rotation.from_euler("y", angle, degrees=True))
#         assert np.all(sig.grid.co >= 0) and np.all(sig.grid.co <= np.pi)
#
#     def test_raises_for_sh_domain_signal(self):
#         sig = make_time_sh(1, 2, 64, 48000)
#         with pytest.raises(ValueError):
#             sig.rotate_space_domain(Rotation.from_euler("z", 45, degrees=True))
#
#
# # ═════════════════════════════════════════════════════════════════════════════
# # rotate_sh_domain()
# # ═════════════════════════════════════════════════════════════════════════════
#
# class TestRotateSHDomain:
#
#     def test_identity_rotation_noop(self):
#         sig = make_time_sh(2, 2, 64, 48000, seed=20)
#         original = sig.data.copy()
#         sig.rotate_sh_domain(Rotation.from_euler("z", 0, degrees=True))
#         np.testing.assert_allclose(sig.data, original, atol=1e-12)
#
#     @pytest.mark.parametrize("sh_order", [1, 2, 3])
#     def test_rotation_is_norm_preserving(self, sh_order):
#         """Wigner-D rotation is unitary — the Frobenius norm must be preserved."""
#         sig = make_time_sh(1, sh_order, 64, 48000, seed=7)
#         norm_before = np.linalg.norm(sig.data)
#         sig.rotate_sh_domain(Rotation.from_euler("zyx", [30, 15, 45], degrees=True))
#         np.testing.assert_allclose(np.linalg.norm(sig.data), norm_before, rtol=1e-8)
#
#     @pytest.mark.parametrize("sh_order", [1, 2, 3])
#     def test_forward_then_inverse_is_identity(self, sh_order):
#         sig = make_time_sh(1, sh_order, 64, 48000, seed=8)
#         original = sig.data.copy()
#         rot = Rotation.from_euler("zyx", [40, 25, -15], degrees=True)
#         sig.rotate_sh_domain(rot)
#         sig.rotate_sh_domain(rot.inv())
#         np.testing.assert_allclose(sig.data, original, atol=1e-10)
#
#     def test_rotation_works_in_freq_domain(self):
#         sig = make_time_sh(1, 2, 64, 48000, seed=12)
#         sig.toFreq()
#         norm_before = np.linalg.norm(sig.data)
#         sig.rotate_sh_domain(Rotation.from_euler("z", 90, degrees=True))
#         np.testing.assert_allclose(np.linalg.norm(sig.data), norm_before, rtol=1e-8)
#
#     def test_raises_for_space_domain_signal(self):
#         sig = make_time_space(1, 8, 64, 48000)
#         with pytest.raises(ValueError):
#             sig.rotate_sh_domain(Rotation.from_euler("z", 45, degrees=True))
#
#     @pytest.mark.parametrize("sh_order", [1, 2])
#     def test_space_rotation_and_sh_rotation_are_consistent(self, sh_order):
#         """
#         Rotating in space domain then toSH must equal toSH then rotating in SH domain.
#         (Fundamental property: Y(R Omega) = D^H Y(Omega) for Wigner-D matrix D.)
#         """
#         grid = from_fibonacci_grid(50)
#         sig = make_sh_subspace_signal(1, sh_order, 32, 48000, grid, seed=99)
#         rot = Rotation.from_euler("z", 45, degrees=True)
#
#         # Path A: rotate grid, then project to SH
#         path_a = sig.copy()
#         path_a.rotate_space_domain(rot)
#         path_a.toSH(sh_order)
#
#         # Path B: project to SH, then rotate coefficients
#         path_b = sig.copy()
#         path_b.toSH(sh_order)
#         path_b.rotate_sh_domain(rot)
#
#         np.testing.assert_allclose(path_a.data, path_b.data, atol=1e-4)
#
#     @pytest.mark.parametrize("angle_deg", [0, 45, 90, 180, 270])
#     def test_360_degree_rotation_is_identity(self, angle_deg):
#         """Cumulative 360° rotation (four 90° steps) must recover the original."""
#         sig = make_time_sh(1, 2, 32, 48000, seed=3)
#         original = sig.data.copy()
#         rot = Rotation.from_euler("z", 90, degrees=True)
#         for _ in range(4):
#             sig.rotate_sh_domain(rot)
#         np.testing.assert_allclose(sig.data, original, atol=1e-10)
#
#
# # ═════════════════════════════════════════════════════════════════════════════
# # Properties
# # ═════════════════════════════════════════════════════════════════════════════
#
# class TestProperties:
#
#     @pytest.mark.parametrize("fs", FS_VALUES)
#     @pytest.mark.parametrize("n_samp", [16, 128, 1000])
#     def test_duration_equals_n_samples_over_fs(self, fs, n_samp):
#         sig = make_time_space(1, 4, n_samp, fs)
#         np.testing.assert_allclose(sig.duration, n_samp / fs)
#
#     def test_duration_is_none_in_freq_domain(self):
#         sig = make_time_space(1, 4, 128, 48000)
#         sig.toFreq()
#         assert sig.duration is None
#
#     def test_n_samples_is_zero_in_freq_domain(self):
#         sig = make_time_space(1, 4, 128, 48000)
#         sig.toFreq()
#         assert sig.n_samples == 0
#
#     @pytest.mark.parametrize("n_ch", [1, 2, 5, 8])
#     def test_n_channels(self, n_ch):
#         assert make_time_space(n_ch, 4, 64, 48000).n_channels == n_ch
#
#     @pytest.mark.parametrize("n_samp", [16, 128, 1000])
#     def test_n_samples_after_zero_pad(self, n_samp):
#         sig = make_time_space(1, 4, n_samp, 48000)
#         sig.zero_pad(n_samp * 2)
#         assert sig.n_samples == n_samp * 2
#
#     @RESAMPLE_PAIRS
#     def test_n_samples_after_resample(self, fs_pair):
#         fs_in, fs_out = fs_pair
#         n = 128
#         sig = make_time_space(1, 4, n, fs_in)
#         sig.resample(fs_out)
#         assert sig.n_samples == int(n * fs_out / fs_in)
