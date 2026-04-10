import numpy as np
import sounddevice as sd

from shroom.acoustics.spatial_signal import SpatialSignal


def play_audio(audio, fs):
    """
    Play audio array using sounddevice.

    Parameters
    ----------
    audio : np.ndarray or SpatialSignal
        Audio data. If np.ndarray, shape should be (n_samples, n_channels) or (n_samples,).
        If SpatialSignal, it extracts the data.
    fs : int
        Sampling frequency.
    """
    # Ensure shape is (n_samples, n_channels) for sounddevice
    if isinstance(audio, SpatialSignal):
        if not audio.is_time:
            raise ValueError("play_audio expect audio to be in time domain")
        if fs is not None:
            if audio.fs != fs:
                raise ValueError("fs must match audio.fs")
        audio = audio.data[:, 0, :].real

    if audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T

    # normalize signal
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.3

    sd.play(audio, fs)
    sd.wait()  # Wait until finished
