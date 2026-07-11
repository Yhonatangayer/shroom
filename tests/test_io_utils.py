"""Tests for file/IO helpers: file_utils, paths, and the pure sofa parsers."""
import os

import numpy as np
import pytest
from scipy.io import wavfile

from shroom.utils.file_utils import load_file, load_wav
from shroom.utils.sofa import convert_sofa_to_radians, is_time, is_sofa_time
from shroom import paths


# ---------------------------------------------------------------------------
# file_utils
# ---------------------------------------------------------------------------

def test_load_file_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_file("some_data.xyz")


def test_load_wav_int16_normalised_to_unit_range(tmp_path):
    p = tmp_path / "int16.wav"
    wavfile.write(str(p), 16000, np.array([0, 32767, -32768], dtype=np.int16))
    audio, fs = load_wav(str(p))
    assert fs == 16000
    assert audio.dtype == np.float32
    assert np.all(np.abs(audio) <= 1.0)


def test_load_wav_int32_normalised(tmp_path):
    p = tmp_path / "int32.wav"
    wavfile.write(str(p), 48000, np.array([0, 2147483647, -2147483648], dtype=np.int32))
    audio, fs = load_wav(str(p))
    assert fs == 48000
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio.max(), 1.0, atol=1e-6)


def test_load_wav_uint8_centered_and_normalised(tmp_path):
    p = tmp_path / "uint8.wav"
    wavfile.write(str(p), 8000, np.array([128, 255, 0], dtype=np.uint8))
    audio, fs = load_wav(str(p))
    assert fs == 8000
    assert audio.dtype == np.float32
    # 128 -> 0 (centre), values stay within [-1, 1]
    np.testing.assert_allclose(audio[0], 0.0, atol=1e-6)
    assert np.all(audio >= -1.0) and np.all(audio <= 1.0)


def test_load_file_dispatches_wav(tmp_path):
    p = tmp_path / "sig.wav"
    wavfile.write(str(p), 16000, np.array([0, 16000, -16000], dtype=np.int16))
    result = load_file(str(p))
    assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# paths.get_default_hrtf_path
# ---------------------------------------------------------------------------

def test_get_default_hrtf_path_returns_existing_file():
    p = paths.get_default_hrtf_path()
    assert os.path.isfile(p)
    assert p.endswith(".sofa")


def test_get_default_hrtf_path_raises_when_missing(monkeypatch):
    monkeypatch.setattr(paths.os.path, "isfile", lambda _p: False)
    with pytest.raises(FileNotFoundError):
        paths.get_default_hrtf_path()


# ---------------------------------------------------------------------------
# sofa pure parsers
# ---------------------------------------------------------------------------

def test_convert_sofa_to_radians_spherical():
    # columns: [azimuth_deg, elevation_deg, radius]
    pos = np.array([[0.0, 90.0, 1.0], [90.0, 0.0, 1.0]])
    az, co = convert_sofa_to_radians(pos, "spherical")
    # el=90 -> colatitude 0 (north pole); el=0 -> colatitude pi/2 (equator)
    np.testing.assert_allclose(co, [0.0, np.pi / 2], atol=1e-12)
    np.testing.assert_allclose(az, [0.0, np.pi / 2], atol=1e-12)


def test_convert_sofa_to_radians_cartesian():
    # +z is straight up (colatitude 0); +x is front (azimuth 0, equator)
    pos = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    az, co = convert_sofa_to_radians(pos, "cartesian")
    np.testing.assert_allclose(co, [0.0, np.pi / 2], atol=1e-9)
    np.testing.assert_allclose(az, [0.0, 0.0], atol=1e-9)


def test_convert_sofa_azimuth_wrapped_to_0_2pi():
    pos = np.array([[-90.0, 0.0, 1.0]])  # -90 deg should wrap to 3*pi/2
    az, _ = convert_sofa_to_radians(pos, "spherical")
    assert 0.0 <= az[0] < 2 * np.pi
    np.testing.assert_allclose(az[0], 3 * np.pi / 2, atol=1e-9)


class _FakeSofa:
    """Minimal stand-in exposing only the attributes the checkers look for."""
    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


@pytest.mark.parametrize("func", [is_time, is_sofa_time])
def test_time_domain_detection(func):
    assert func(_FakeSofa(Data_IR=np.zeros(1))) is True
    assert func(_FakeSofa(Data_Real=np.zeros(1))) is False


@pytest.mark.parametrize("func", [is_time, is_sofa_time])
def test_time_domain_detection_raises_without_data(func):
    with pytest.raises(ValueError):
        func(_FakeSofa())
