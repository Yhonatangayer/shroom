import pytest
import os
import numpy as np
from shroom.utils.file_utils import load_file
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.utils.dsp_utils import (
    is_signal_frequency_symmetric,
    is_signal_frequency_sh_valid,
)

# Path to HRTF file (adjust if needed)
HRTF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data/sofa_hrtfs", "mit_kemar_normal_pinna.sofa"
)


@pytest.mark.skipif(not os.path.exists(HRTF_PATH), reason="HRTF file not found")
def test_hrtf_loading():
    """Test loading and validity of the HRTF file."""
    hrtf = load_file(HRTF_PATH)

    # 1. Check Type
    assert isinstance(hrtf, SpatialSignal)

    # 2. Check Domain
    assert hrtf.is_time
    assert hrtf.is_space  # HRTF is defined on a spatial grid

    # 3. Check Shape
    # Expected: (Channels=2 ears, Grid=N_dirs, Time=N_samples)
    assert hrtf.n_channels == 2
    assert hrtf.grid is not None
    assert hrtf.data.shape[1] == hrtf.grid.n_points
    assert hrtf.n_samples > 0

    print(f"HRTF Shape: {hrtf.data.shape}")
    print(f"Grid Points: {hrtf.grid.n_points}")

    # 4. Check Validity (Symmetry)
    # FFT along time axis
    hrtf_freq = np.fft.fft(hrtf.data, axis=-1)

    assert is_signal_frequency_symmetric(hrtf_freq, freq_axis=-1)


def test_hrtf_sh_conversion():
    """Test converting HRTF to SH domain."""
    if not os.path.exists(HRTF_PATH):
        pytest.skip("HRTF file not found")

    hrtf = load_file(HRTF_PATH)

    # Convert to SH
    sh_order = 3
    hrtf.toSH(N_sp=sh_order)

    assert hrtf.is_sh
    assert hrtf.is_time

    # Check shape: (Ears, SH_Coeffs, Time)
    n_sh = (sh_order + 1) ** 2
    assert hrtf.data.shape[1] == n_sh

    # Check SH Validity (DC of higher orders should be small/zero)
    # Note: HRTF measurements might have DC noise, so we use a lenient tolerance
    # or just check symmetry.

    hrtf_freq = np.fft.fft(hrtf.data, axis=-1)

    # Check symmetry for SH coeffs
    assert is_signal_frequency_sh_valid(hrtf_freq, freq_axis=-1)
