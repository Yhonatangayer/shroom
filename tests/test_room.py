import warnings

import pytest
import numpy as np
import pyroomacoustics as pra
import os
from shroom.acoustics.room import Room
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.acoustics.processors import BinauralDecoder
from shroom.utils.dsp_utils import is_signal_frequency_sh_valid
from shroom.paths import DEFAULT_WAV_PATH


@pytest.fixture
def basic_room():
    """A simple room fixture for testing."""
    return Room(dimensions=[5, 4, 3], absorption=0.2, fs=48000, sh_order=3)


@pytest.fixture
def specific_room():
    room = Room(
        dimensions=[8.0, 6.0, 4.0],
        absorption=0.9,
        max_ism_order=20,
        fs=48000.0,
        sh_order=3,
    )
    room.add_source([4.0, 3.0, 1.7], signal=DEFAULT_WAV_PATH)
    room.set_receiver([2.6, 4.4, 1.7])
    return room


def test_room_creation(basic_room):
    """Test that the Room object is created with correct parameters."""
    assert basic_room.dimensions.tolist() == [5, 4, 3]
    assert basic_room.fs == 48000
    assert basic_room.pra_room is not None


def test_add_source_and_receiver(basic_room):
    """Test adding a source and setting a receiver."""
    basic_room.add_source([1, 1, 1.5])
    basic_room.set_receiver([2, 2, 1.5])

    assert len(basic_room.sources) == 1
    assert basic_room.sources[0]["position"].tolist() == [1, 1, 1.5]
    assert basic_room.receiver_position.tolist() == [2, 2, 1.5]


def test_compute_arir(basic_room):
    """Test ARIR computation."""
    basic_room.add_source([1, 1, 1.5])
    basic_room.set_receiver([2, 2, 1.5])

    sh_order = 3
    arirs = basic_room.compute_arir()

    assert isinstance(arirs, list)
    assert len(arirs) == 1

    arir = arirs[0]
    assert isinstance(arir, SpatialSignal)
    assert arir.is_sh
    assert arir.is_time
    assert arir.data.shape[0] == 1  # n_channels (from SpatialSignal perspective)
    assert arir.data.shape[1] == (sh_order + 1) ** 2  # n_grid (SH coeffs)
    assert arir.data.dtype == np.complex128
    assert np.sum(np.abs(arir.data)) > 0  # Check for non-zero energy

    # Check SH validity (DC of n>0 should be 0)
    # arir.data is (1, SH, Time)
    # FFT to check freq domain
    arir_freq = np.fft.fft(arir.data, axis=-1)
    # Pass (SH, Freq) -> arir_freq[0, :, :]
    assert is_signal_frequency_sh_valid(arir_freq[0, :, :], freq_axis=-1, sh_axis=0)


def test_compute_amb(basic_room):
    """Test Ambisonics simulation with a source signal."""
    signal = np.random.randn(48000)  # 1 second of noise
    basic_room.add_source([1, 1, 1.5], signal=signal)
    basic_room.set_receiver([2, 2, 1.5])

    sh_order = 3
    amb_signal = basic_room.compute_amb()

    assert isinstance(amb_signal, SpatialSignal)
    assert amb_signal.is_sh
    assert amb_signal.is_time
    assert amb_signal.data.shape[1] == (sh_order + 1) ** 2
    assert amb_signal.n_samples > len(signal)  # Convolution makes it longer
    assert np.sum(np.abs(amb_signal.data)) > 0

    # Check SH validity
    amb_freq = np.fft.fft(amb_signal.data, axis=-1)
    assert is_signal_frequency_sh_valid(amb_freq[0, :, :], freq_axis=-1, sh_axis=0)


def test_binaural_decoding(basic_room):
    """Test binaural decoding using the Processor."""
    # 1. Add source and receiver
    signal = np.random.randn(1000)
    basic_room.add_source([1, 1, 1.5], signal=signal)
    basic_room.set_receiver([2, 2, 1.5])

    # 2. Compute Ambisonics
    sh_order = 3
    amb_signal = basic_room.compute_amb()

    # 3. Create a mock HRTF in SH domain
    n_sh = (sh_order + 1) ** 2
    n_ears = 2
    hrtf_len = 512

    # Mock HRTF data: (n_ears, n_sh_coeffs, n_samples)
    mock_hrtf_data = np.zeros((n_ears, n_sh, hrtf_len), dtype=np.complex128)
    mock_hrtf_data[0, 0, 10] = 1  # Left ear, W channel impulse
    mock_hrtf_data[1, 0, 12] = 1  # Right ear, W channel impulse

    mock_hrtf = SpatialSignal(
        data=mock_hrtf_data,
        fs=basic_room.fs,
        is_time=True,
        is_space=False,  # SH domain
        grid=None,
    )

    # 4. Decode
    decoder = BinauralDecoder(mock_hrtf, sh_order=3, output_format="SpatialSignal")
    binaural_output = decoder.process(amb_signal)

    # Check output
    # BinauralDecoder returns np.ndarray (Ears, Time)
    assert isinstance(binaural_output, SpatialSignal)
    assert binaural_output.data.shape[0] == n_ears
    assert binaural_output.data.shape[1] == 1
    assert binaural_output.data.shape[2] > len(signal)
    assert np.sum(np.abs(binaural_output.data)) > 0


def _wall_absorption(room):
    """Energy absorption coefficient applied to the room's first wall."""
    return float(room.pra_room.walls[0].absorption[0])


@pytest.mark.parametrize("coeff", [0.1, 0.2, 0.5, 0.8])
def test_absorption_energy_mode_applied_directly(coeff):
    """Default 'energy' mode applies the coefficient as energy absorption directly."""
    room = Room(dimensions=[5, 4, 3], absorption=coeff, fs=48000, sh_order=3)
    assert _wall_absorption(room) == pytest.approx(coeff, abs=1e-6)


@pytest.mark.parametrize("coeff", [0.1, 0.2, 0.5, 0.8])
def test_absorption_energy_mode_matches_material(coeff):
    """absorption=a (energy) must equal materials=pra.Material(a)."""
    room_abs = Room(dimensions=[5, 4, 3], absorption=coeff, fs=48000, sh_order=3)
    room_mat = Room(
        dimensions=[5, 4, 3], materials=pra.Material(coeff), fs=48000, sh_order=3
    )
    assert _wall_absorption(room_abs) == pytest.approx(
        _wall_absorption(room_mat), abs=1e-6
    )


@pytest.mark.parametrize("coeff", [0.1, 0.2, 0.5, 0.8])
def test_absorption_legacy_mode_reproduces_old_behavior(coeff):
    """'legacy' mode reproduces the pre-0.2.0 1-(1-a)**2 conversion."""
    room = Room(
        dimensions=[5, 4, 3],
        absorption=coeff,
        absorption_mode="legacy",
        fs=48000,
        sh_order=3,
    )
    expected = 1.0 - (1.0 - coeff) ** 2
    assert _wall_absorption(room) == pytest.approx(expected, abs=1e-6)


def test_absorption_dict_energy_mode():
    """Per-wall dict absorption is applied directly in energy mode."""
    walls = {
        "east": 0.1,
        "west": 0.2,
        "north": 0.3,
        "south": 0.4,
        "ceiling": 0.5,
        "floor": 0.6,
    }
    room = Room(dimensions=[5, 4, 3], absorption=walls, fs=48000, sh_order=3)
    applied = {w.name: float(w.absorption[0]) for w in room.pra_room.walls}
    for name, coeff in walls.items():
        assert applied[name] == pytest.approx(coeff, abs=1e-6)


def test_absorption_no_deprecation_warning():
    """Constructing a Room with a float absorption must not emit a DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        Room(dimensions=[5, 4, 3], absorption=0.8, fs=48000, sh_order=3)


def test_invalid_absorption_mode_raises():
    with pytest.raises(ValueError, match="absorption_mode"):
        Room(
            dimensions=[5, 4, 3],
            absorption=0.5,
            absorption_mode="bogus",
            fs=48000,
            sh_order=3,
        )


def test_compare_arir_generation(specific_room):
    path = os.path.join(os.path.dirname(__file__), "hnm_ref.npz")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")

    specific_room._remove_dc = False
    hnm_ref = np.load(path)["data"][: (3 + 1) ** 2, :]
    hnm = specific_room.arirs[0].data[0, ...]
    max_diff = np.abs(hnm_ref - hnm).max()
    assert np.allclose(
        hnm_ref, hnm
    ), f"arir is different than old simulation arir, with max diff ({max_diff})"


def test_compare_ambisonics_generation(specific_room):
    path = os.path.join(os.path.dirname(__file__), "anm_ref.npz")
    if not os.path.exists(path):
        pytest.skip(f"Reference file not found: {path}")

    specific_room._remove_dc = False
    amb_ref = np.load(path)["data"][: (3 + 1) ** 2, :]
    amb = specific_room.compute_amb()
    amb = amb.data[0, ...]
    max_diff = np.abs(amb_ref - amb).max()
    assert np.allclose(
        amb_ref, amb
    ), f"arir is different than old simulation arir, with max diff ({max_diff})"


# ---------------------------------------------------------------------------
# Octave-band absorption
# ---------------------------------------------------------------------------

OCTAVE_CENTERS = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def _band_material(coeffs):
    """pra.Material from eight octave-band energy absorption coefficients."""
    return pra.Material(
        energy_absorption={"coeffs": list(coeffs), "center_freqs": OCTAVE_CENTERS}
    )


def _arir(**kwargs):
    room = Room(dimensions=[6, 5, 3], fs=48000, **kwargs)
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return room.compute_arir()[0].data[0]


def test_flat_multiband_material_matches_scalar_absorption():
    """A material with 8 identical coefficients must reproduce the scalar-a response.

    Regression test: the octave-band amplitudes used to be summed into a single
    broadband delta, which made a multi-band material ~n_bands times too loud.
    """
    coeff = 0.15
    scalar = _arir(absorption=coeff, max_ism_order=6, sh_order=2)
    multi = _arir(materials=_band_material([coeff] * 8), max_ism_order=6, sh_order=2)

    n = min(scalar.shape[1], multi.shape[1])
    err = np.abs(multi[:, :n] - scalar[:, :n]).max() / np.abs(scalar).max()
    assert err < 1e-5, f"multi-band material deviates from scalar by {err:.2e} of peak"


def test_multiband_material_preserves_frequency_dependence():
    """Strong HF absorption must show up as an HF rolloff in the late response."""

    def hf_lf_ratio(x):
        late = np.real(x[0])[len(x[0]) // 2 :]
        spectrum = np.abs(np.fft.rfft(late)) ** 2
        freqs = np.fft.rfftfreq(late.size, 1 / 48000)
        return spectrum[freqs > 4000].sum() / spectrum[freqs < 1000].sum()

    flat = _arir(materials=_band_material([0.15] * 8), max_ism_order=8, sh_order=1)
    hf_absorbing = _arir(
        materials=_band_material([0.05, 0.05, 0.05, 0.10, 0.40, 0.70, 0.85, 0.90]),
        max_ism_order=8,
        sh_order=1,
    )
    assert hf_lf_ratio(hf_absorbing) < 0.01 * hf_lf_ratio(flat)


def test_scalar_absorption_stays_single_band():
    """Scalar absorption must keep the unfiltered single-band path (bit-exact output)."""
    room = Room(dimensions=[6, 5, 3], absorption=0.15, max_ism_order=6, sh_order=1)
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])
    room.pra_room.image_source_model()
    assert room.pra_room.sources[0].damping.shape[0] == 1


# ---------------------------------------------------------------------------
# Randomized ISM
# ---------------------------------------------------------------------------


def test_rand_ism_defaults_on_and_is_deterministic():
    """Randomized ISM is on by default, and the default seed keeps it reproducible."""
    room = Room(dimensions=[6, 5, 3], absorption=0.15, max_ism_order=4, sh_order=1)
    assert room.use_rand_ism is True
    assert room.seed == 0
    assert room.pra_room.simulator_state["random_ism_needed"] is True
    assert np.array_equal(
        _arir(absorption=0.15, max_ism_order=4, sh_order=1),
        _arir(absorption=0.15, max_ism_order=4, sh_order=1),
    )


def test_rand_ism_can_be_disabled_for_exact_image_positions():
    room = Room(
        dimensions=[6, 5, 3], absorption=0.15, max_ism_order=4, use_rand_ism=False
    )
    assert room.pra_room.simulator_state["random_ism_needed"] is False


def test_unseeded_rand_ism_varies_between_runs():
    a = _arir(absorption=0.15, max_ism_order=6, sh_order=1, seed=None)
    b = _arir(absorption=0.15, max_ism_order=6, sh_order=1, seed=None)
    assert not np.array_equal(a, b)


def test_use_rand_ism_displaces_images():
    """Randomized ISM must jitter image positions and break delay degeneracy."""

    def images_and_delays(**kwargs):
        room = Room(dimensions=[6, 5, 3], absorption=0.15, max_ism_order=8, **kwargs)
        room.add_source([4.0, 4.0, 1.5])
        room.set_receiver([2.0, 2.0, 1.5])
        room.pra_room.image_source_model()
        images = room.pra_room.sources[0].images
        dist = np.linalg.norm(images - room.receiver_position[:, None], axis=0)
        delays = np.round(dist / room.pra_room.c * room.fs).astype(int)
        return images, np.unique(delays).size

    exact_images, exact_unique = images_and_delays(use_rand_ism=False)
    rand_images, rand_unique = images_and_delays(use_rand_ism=True, max_rand_disp=0.08)

    assert exact_images.shape == rand_images.shape
    displacement = np.abs(rand_images - exact_images).max()
    assert 0 < displacement <= 0.08 + 1e-9
    assert rand_unique > exact_unique


def test_seed_makes_rand_ism_reproducible():
    """The same seed must reproduce the ARIR exactly; a different seed must not."""

    def run(seed):
        room = Room(
            dimensions=[6, 5, 3],
            absorption=0.15,
            max_ism_order=6,
            sh_order=1,
            use_rand_ism=True,
            seed=seed,
        )
        room.add_source([4.0, 4.0, 1.5])
        room.set_receiver([2.0, 2.0, 1.5])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return room.compute_arir()[0].data[0]

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_seed_does_not_disturb_numpy_random_state():
    np.random.seed(1234)
    expected = np.random.rand(3)

    np.random.seed(1234)
    room = Room(
        dimensions=[6, 5, 3],
        absorption=0.15,
        max_ism_order=4,
        sh_order=0,
        use_rand_ism=True,
        seed=99,
    )
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        room.compute_arir()

    assert np.array_equal(np.random.rand(3), expected)


def test_rand_ism_validation():
    with pytest.raises(TypeError, match="use_rand_ism"):
        Room(dimensions=[5, 4, 3], absorption=0.2, use_rand_ism="yes")
    with pytest.raises(ValueError, match="max_rand_disp"):
        Room(dimensions=[5, 4, 3], absorption=0.2, max_rand_disp=-0.1)
    with pytest.raises(TypeError, match="seed"):
        Room(dimensions=[5, 4, 3], absorption=0.2, seed="abc")


# ---------------------------------------------------------------------------
# ray_tracing honesty
# ---------------------------------------------------------------------------


def test_ray_tracing_raises():
    """ray_tracing=True must fail loudly rather than be silently ignored."""
    with pytest.raises(NotImplementedError, match="ray_tracing=True is not supported"):
        Room(dimensions=[5, 4, 3], absorption=0.2, ray_tracing=True)


def test_ray_tracing_off_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        Room(dimensions=[5, 4, 3], absorption=0.2, ray_tracing=False)


# ---------------------------------------------------------------------------
# ISM coverage guardrail
# ---------------------------------------------------------------------------


def test_ism_coverage_reports_truncated_tail():
    room = Room(dimensions=[6, 5, 3], absorption=0.15, max_ism_order=20, sh_order=0)
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])

    estimate = room.ism_coverage()
    assert estimate["exact_length"] is False  # ISM not run yet
    assert estimate["mean_absorption"] == pytest.approx(0.15, abs=1e-5)
    assert estimate["t60_eyring"] == pytest.approx(0.708, rel=0.02)

    with pytest.warns(UserWarning, match="covers only"):
        room.compute_arir()

    info = room.ism_coverage()
    assert info["exact_length"] is True
    assert info["coverage"] < 1.0
    assert info["required_order"] > 20


def test_ism_coverage_quiet_when_order_is_sufficient():
    room = Room(dimensions=[6, 5, 3], absorption=0.6, max_ism_order=20, sh_order=0)
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        room.compute_arir()
    assert room.ism_coverage()["coverage"] > 1.0


def test_ism_coverage_anechoic_room_is_not_flagged():
    """A fully absorbing room has no reverberation to truncate."""
    room = Room(dimensions=[6, 5, 3], absorption=1.0, max_ism_order=5, sh_order=0)
    room.add_source([4.0, 4.0, 1.5])
    room.set_receiver([2.0, 2.0, 1.5])
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        room.compute_arir()
