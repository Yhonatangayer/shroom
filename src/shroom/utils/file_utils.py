import numpy as np
from typing import Tuple
from shroom.utils.sofa import load_sofa
from scipy.io import wavfile


def load_file(file_path: str):
    """
    Load a file based on its extension.
    Supports .sofa, .wav, .npz.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    Loaded object (SpatialSignal, tuple, dict, etc.)
    """
    suffix_to_format = {
        ".sofa": load_sofa,
        ".wav": load_wav,
        ".npz": np.load,
    }
    suffix = "." + file_path.split(".")[-1]
    if suffix in suffix_to_format:
        return suffix_to_format[suffix](file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_wav(wav_path: str) -> Tuple[int, np.array]:
    """Load a WAV file."""
    fs, audio = wavfile.read(wav_path)

    # Normalize to float [-1, 1] if int
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0

    return audio, fs
