import pytest
import numpy as np
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
    decoder = BinauralDecoder(mock_hrtf, sh_order=3, output_format='SpatialSignal')
    binaural_output = decoder.process(amb_signal)

    # Check output
    # BinauralDecoder returns np.ndarray (Ears, Time)
    assert isinstance(binaural_output, SpatialSignal)
    assert binaural_output.data.shape[0] == n_ears
    assert binaural_output.data.shape[1] == 1
    assert binaural_output.data.shape[2] > len(signal)
    assert np.sum(np.abs(binaural_output.data)) > 0


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
