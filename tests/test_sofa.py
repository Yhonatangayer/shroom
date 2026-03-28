import pytest
import os
import glob
import numpy as np
from shroom.utils.sofa import load_sofa
from shroom.acoustics.spatial_signal import SpatialSignal

# Determine project root relative to this test file
# tests/test_sofa.py -> ../ -> project root
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

def get_sofa_files():
    """Helper to get all .sofa files from the data directories."""
    sofa_arrays = os.path.join(DATA_DIR, "sofa_arrays")
    sofa_hrtfs = os.path.join(DATA_DIR, "sofa_hrtfs")

    array_files = glob.glob(os.path.join(sofa_arrays, "*.sofa"))
    hrtf_files = glob.glob(os.path.join(sofa_hrtfs, "*.sofa"))

    all_files = array_files + hrtf_files
    return all_files
    # return hrtf_files

@pytest.mark.parametrize("sofa_path", get_sofa_files())
def test_load_sofa_files(sofa_path):
    """
    Test loading all available SOFA files and validating they result in a valid SpatialSignal.
    """
    print(f"Testing SOFA file: {sofa_path}")

    if not os.path.exists(sofa_path):
        pytest.skip(f"File not found: {sofa_path}")

    try:
        signal = load_sofa(sofa_path)
    except Exception as e:
        pytest.fail(f"Failed to load SOFA file {sofa_path}: {e}")

    # Validate that the result is a SpatialSignal
    assert isinstance(signal, SpatialSignal), f"Loaded object from {sofa_path} is not a SpatialSignal"

    # Validate basic properties
    assert signal.fs > 0, f"Invalid sampling rate in {sofa_path}"
    assert signal.n_channels > 0, f"No channels found in {sofa_path}"

    # Check domain flags
    if signal.is_time:
        assert signal.n_samples > 0, f"Time domain signal has 0 samples in {sofa_path}"
        assert not signal.is_freq
    else:
        assert signal.is_freq
        assert not signal.is_time

    # Check spatial domain
    assert signal.is_space, f"Loaded SOFA signal from {sofa_path} should be in Space domain"
    assert not signal.is_sh

    # Check grid
    assert signal.grid is not None, f"Grid is missing for {sofa_path}"

    # Check data shape consistency with grid
    # SpatialSignal data shape is (Channels, GridPoints, Time/Freq)
    # For SOFA, usually Channels corresponds to Emitters/Receivers and GridPoints to Measurements
    assert signal.data.shape[1] == signal.grid.n_points, \
        f"Grid points ({signal.grid.n_points}) mismatch data shape ({signal.data.shape}) in {sofa_path}"

    # Check data validity (no NaNs or Infs)
    assert np.all(np.isfinite(signal.data)), f"Data contains NaNs or Infs in {sofa_path}"
