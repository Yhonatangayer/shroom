import pytest
import numpy as np
from shroom.acoustics.processors import (
    ProcessorChain,
    ArrayDecoder,
    ASMEncoder,
    BinauralDecoder,
)
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_spaudiopy_grid, from_fibonacci_grid
# from spaudiopy.grids import load_lebedev


@pytest.fixture
def mock_amb_signal():
    """Create a mock Ambisonics signal (SH domain)."""
    fs = 16000
    sh_order = 1
    n_sh = (sh_order + 1) ** 2
    n_samples = 1000

    # (SH, 1, Time)
    data = np.zeros((1, n_sh, n_samples))
    data[0, 0, 0] = 1.0  # Impulse in W channel

    return SpatialSignal(
        data=data, fs=fs, is_time=True, is_space=False, grid=None  # SH
    )


@pytest.fixture
def array_setup():
    """Create array and ASM instance."""
    fs = 16000
    radius = 0.1
    n_mics = 6

    mics_grid = sphereicalGrid(
        az=np.linspace(0, 2 * np.pi, n_mics, endpoint=False),
        co=np.full(n_mics, np.pi / 2),
    )
    source_grid = from_fibonacci_grid(50)

    array = SphericalArray(
        fs=fs,
        duration=0.01,
        r_sphere=radius,
        r_mics=np.full(n_mics, radius),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=1,
        convert_to_time=False,
    )
    array_sh_time = array.copy()
    array_sh_time.toTime()
    array_sh_time.toSH(1)

    asm = ASM(sh_order=1, array=array, fs=fs, duration=0.01)

    return array_sh_time, asm


@pytest.fixture
def mock_hrtf():
    """Create a mock HRTF in SH domain."""
    fs = 16000
    sh_order = 1
    n_sh = (sh_order + 1) ** 2
    n_ears = 2
    n_samples = 128

    data = np.zeros((n_ears, n_sh, n_samples))
    data[0, 0, 0] = 1.0  # Left ear W
    data[1, 0, 0] = 1.0  # Right ear W

    # Wrap in SpatialSignal (Channels=Ears, Grid=SH, Time)
    # Wait, SpatialSignal usually expects (Channels, Grid, Time).
    # If it's SH, 'Grid' dimension is SH coeffs?
    # Yes, usually.

    return SpatialSignal(
        data=data,
        fs=fs,
        is_time=True,
        is_space=False,
        grid=None,  # SH domain doesn't need grid object usually
    )


def test_processor_chain(mock_amb_signal, array_setup, mock_hrtf):
    """Test the full chain: Amb -> Array -> ASM -> Binaural."""
    array, asm = array_setup

    # 1. Define Processors
    array_decoder = ArrayDecoder(array)
    asm_encoder = ASMEncoder(asm)
    binaural_decoder = BinauralDecoder(
        mock_hrtf, sh_order=1, output_format="SpatialSignal"
    )

    # 2. Create Chain
    chain = ProcessorChain([array_decoder, asm_encoder, binaural_decoder])

    # 3. Process
    binaurals = chain.process(mock_amb_signal)

    # 4. Verify
    assert isinstance(binaurals, SpatialSignal)
    assert binaurals.is_time
    assert binaurals.is_space == False  # Binaural
    assert binaurals.data.shape[0] == 2  # Binaural
    assert binaurals.data.shape[1] == 1
    assert binaurals.data.shape[2] > 0


def test_processor_chain_numerical(mock_amb_signal, array_setup, mock_hrtf):
    """Chain output must match sequential processor application numerically."""
    array, asm = array_setup

    array_decoder = ArrayDecoder(array)
    asm_encoder = ASMEncoder(asm)
    binaural_decoder = BinauralDecoder(
        mock_hrtf, sh_order=1, output_format="SpatialSignal"
    )

    # Sequential processing
    mic_signals = array_decoder.process(mock_amb_signal)
    amb_encoded = asm_encoder.process(mic_signals)
    binaural_seq = binaural_decoder.process(amb_encoded)

    # Chain processing (fresh chain so unified kernel is computed from scratch)
    chain = ProcessorChain([
        ArrayDecoder(array),
        ASMEncoder(asm),
        BinauralDecoder(mock_hrtf, sh_order=1, output_format="SpatialSignal"),
    ])
    binaural_chain = chain.process(mock_amb_signal)

    # Trim to the shorter length to allow for edge differences
    T = min(binaural_seq.data.shape[2], binaural_chain.data.shape[2])
    seq_data = binaural_seq.data[:, :, :T].real
    chain_data = binaural_chain.data[:, :, :T].real

    np.testing.assert_allclose(chain_data, seq_data, atol=1e-6,
                               err_msg="ProcessorChain output differs from sequential processing")
