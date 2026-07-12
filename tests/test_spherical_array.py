import pytest
import numpy as np
from shroom.utils.dsp_utils import (
    is_signal_frequency_sh_valid,
    is_signal_frequency_symmetric,
    is_sh_valid,
)


@pytest.fixture
def basic_array(make_spherical_array):
    """A basic rigid spherical array (equator mics, Fibonacci sources)."""
    return make_spherical_array(
        n_mics=6,
        radius=0.1,
        fs=48000,
        duration=0.01,
        source_points=974,
        sh_order_for_sm_calc=14,
        convert_to_time=False,
    )


def test_array_initialization(basic_array):
    """Test array initialization and properties."""
    assert basic_array.fs == 48000
    assert basic_array.n_channels == 6  # 6 mics
    assert basic_array.is_space  # It's a spatial signal (mic signals)
    assert basic_array.grid.n_points == 974  # Lebedev 53 -> 974 points


def test_array_data_shape(basic_array):
    """Test the shape of the generated array manifold/data."""
    n_mics = 6
    n_sources = 974  # Lebedev degree 53 -> 974 points

    assert basic_array.data.shape[0] == n_mics
    assert basic_array.data.shape[1] == n_sources
    assert basic_array.data.shape[2] > 0  # Time samples


def test_bn_not_zeroed_at_sub_unit_k():
    """Regression: Bn n>=1 must not be zeroed at k < 1 rad/m (non-DC bins).

    With fs=48000 and nFFT=1024, index 1 is f=46.875 Hz → k=0.858 rad/m < 1.
    The old code used `k < 1` as the zero-mask, incorrectly killing n>=1 modes
    at this bin and making the steering matrix rank-1 there.
    """
    from shroom.acoustics.physics import _compute_bn_diagonal

    fs, nFFT = 48000, 1024
    pos_freqs = np.fft.rfftfreq(nFFT, 1 / fs)
    k = 2 * np.pi * pos_freqs / 343.0

    assert k[1] < 1.0, "test assumption: k[1] must be < 1 rad/m for this setup"

    bn = _compute_bn_diagonal(
        N=5, k=k, a=0.1, r_m=0.1,
        sphere_type="rigid", source_type="plane_wave", apply_damping=False,
    )
    # Row 1 corresponds to the first m=0 component of n=1 (ACN index 1)
    assert np.abs(bn[1, 1]) > 1e-6, (
        f"Bn n=1 at k[1]={k[1]:.4f} rad/m should be non-zero, got {bn[1, 1]}"
    )


def test_array_signal_validity(basic_array):
    """Test signal validity (symmetry, space/SH properties)."""
    # Data is (Mics, Sources, Freq)
    assert basic_array.is_freq
    assert basic_array.is_space
    # FFT along time axis
    data_space_freq = basic_array.data

    # 1. Check Symmetry (Real time signal)
    # Check for a single channel/source combination
    assert is_signal_frequency_symmetric(data_space_freq, freq_axis=-1)

    # 2. Check Space Validity (should be same as symmetric for space domain)
    # assert is_signal_frequency_space_valid(data_space_freq, freq_axis=-1)

    # 3. Check SH Validity
    array_sh = basic_array.copy()
    array_sh.toSH(N_sp=1)

    # 3.1
    pY = basic_array.grid.pinvY(1)
    Y = basic_array.grid.Y(1)
    assert is_sh_valid(Y, sh_axis=1)
    assert is_sh_valid(pY, sh_axis=0)

    # FFT the SH data
    assert array_sh.is_sh
    assert array_sh.is_freq
    data_sh_freq = array_sh.data

    # Check SH validity for a specific source direction (e.g. source 0)
    assert is_signal_frequency_sh_valid(data_sh_freq, freq_axis=2, sh_axis=1)


# ---------------------------------------------------------------------------
# Radial-function damping (physics.py)
#
# These tests pin down the radial-function damping after the per-order sigmoid
# "order mask" 1/(1+exp(n-(ka+1))) was removed. The mask laid a staircase of N
# sharp spectral edges along frequency, which rings in the time domain and is
# heard as a duplicate/ghost image for long (high-DURATION) radial filters. It
# was replaced by — and was always redundant with — a smooth Wiener-style
# magnitude knee |b_n|^2 / (|b_n|^2 + limit^2) that suppresses the same
# numerically-insignificant coefficients without any order/frequency gate.
#
# Filter encoders (ASM/BSM/AA-MagLS) now consume this array model, so their
# precision tests are deliberately decoupled (synthetic steering matrix) and the
# damping contract is verified here instead.
# ---------------------------------------------------------------------------

DAMPING_LIMIT = 1e-4  # must match `limit` in physics._apply_magnitude_damping


def test_magnitude_damping_is_wiener_knee():
    """Damping is exactly the magnitude knee: out = b * |b|^2/(|b|^2+limit^2)."""
    from shroom.acoustics.physics import _apply_magnitude_damping

    bn = np.array(
        [
            [1.0 + 0j, 5.0 - 2.0j, 1e-2 + 0j],  # significant (|b| >> limit)
            [1e-3 + 0j, 1e-6 + 0j, 0.0 + 0j],   # mid / tiny / exact zero
        ]
    )
    out = _apply_magnitude_damping(bn.copy())
    gain = np.abs(bn) ** 2 / (np.abs(bn) ** 2 + DAMPING_LIMIT**2)
    np.testing.assert_allclose(out, bn * gain, atol=1e-15)

    # Significant coefficients pass through essentially untouched...
    assert np.abs(out[0, 0]) > 0.999 * np.abs(bn[0, 0])
    # ...while coefficients far below the limit are strongly suppressed.
    assert np.abs(out[1, 1]) < 1e-3 * np.abs(bn[1, 1])


def test_damping_applies_only_magnitude_knee_no_order_mask():
    """Regression: damping depends only on |b_n|, never on order n or frequency.

    Equivalent to asserting the removed order mask is truly gone — the damped
    radial functions equal the undamped ones times the pure magnitude knee at
    every non-DC bin (DC carries a separate real/zero mask).
    """
    from shroom.acoustics.physics import _compute_bn_diagonal

    fs, nFFT = 16000, 256
    k = 2 * np.pi * np.fft.rfftfreq(nFFT, 1 / fs) / 343.0
    kw = dict(N=6, k=k, a=0.05, r_m=0.05, sphere_type="rigid",
              source_type="plane_wave")

    bn_on = _compute_bn_diagonal(**kw, apply_damping=True)
    bn_off = _compute_bn_diagonal(**kw, apply_damping=False)

    gain = np.abs(bn_off) ** 2 / (np.abs(bn_off) ** 2 + DAMPING_LIMIT**2)
    # Exclude DC (column 0): n>=1 is zeroed and n=0 forced real there.
    np.testing.assert_allclose(bn_on[:, 1:], (bn_off * gain)[:, 1:],
                               atol=1e-12, rtol=1e-9)


def test_damping_never_amplifies():
    """The knee can only attenuate (gain <= 1): it must never boost a mode."""
    from shroom.acoustics.physics import _compute_bn_diagonal

    fs, nFFT = 16000, 256
    k = 2 * np.pi * np.fft.rfftfreq(nFFT, 1 / fs) / 343.0
    kw = dict(N=6, k=k, a=0.05, r_m=0.05, sphere_type="rigid",
              source_type="plane_wave")

    bn_on = _compute_bn_diagonal(**kw, apply_damping=True)
    bn_off = _compute_bn_diagonal(**kw, apply_damping=False)

    assert np.all(np.abs(bn_on) <= np.abs(bn_off) + 1e-12)
