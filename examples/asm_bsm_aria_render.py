import numpy as np
import matplotlib.pyplot as plt
from shroom.acoustics.room import Room
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.processors import (
    ProcessorChain,
    ArrayDecoder,
    ASMEncoder,
    BinauralDecoder,
    BSMEncoder,
)
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_spaudiopy_grid, from_fibonacci_grid
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.utils.file_utils import load_file
from shroom.utils.sofa import load_sofa
from devutils.sound import play_audio
from scipy.spatial.transform import Rotation
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

ARIA_PATH = "/Users/yhonag/repos/py/shroom/data/sofa_arrays/aria_atfs_fixed.sofa"

# --- Configuration ---
FS = 48000.0
DURATION = 0.008
MAGLS_CUTOFF = 1400.0
IS_MAGLS_HRTF = True
SH_ORDER = 20
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 10
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [0, 0, 0]


def main():
    print("--- Starting Binaural Room Simulation (ASM Chain) ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS, absorption=ABSORPTION, max_ism_order=MAX_ISM_ORDER, fs=FS
    )
    print(f"Room initialized: {ROOM_DIMS}m, Absorption: {ABSORPTION}")

    array = load_sofa(ARIA_PATH)
    array.resample(FS)

    # Prepare array for ASM (Frequency Domain)
    array.toFreq()
    array_time_sh = array.copy()
    array_time_sh.toTime()
    array_time_sh.toSH(1)

    # 3. Setup ASM
    asm = ASM(sh_order=1, array=array, fs=FS, duration=DURATION)

    # Setup BSM
    n_fft = array.data.shape[-1]
    array_freq_space = array.copy()  # already in freq domain
    hrtf_space = load_file(DEFAULT_HRTF_PATH)
    hrtf_space.resample(desired_fs=FS)
    hrtf_copy = hrtf_space.copy()
    hrtf_space.zero_pad(n_fft)
    hrtf_space.toFreq()
    hrtf_space.toSH(30)
    hrtf_space.toSpace(array.grid)
    array_space = array.copy()  # (M, Q, F) freq+space on native grid

    print(f"Computing BSM + MagLS (cutoff={MAGLS_CUTOFF} Hz)...")
    bsm = BSM(
        array=array_space,
        hrtf=hrtf_space,
        use_magls=True,
        magls_cutoff_frequency=MAGLS_CUTOFF,
        fs=FS,
        duration=DURATION,
    )

    array_sh_bsm = array.copy()
    array_sh_bsm.toTime()
    array_sh_bsm.toSH(SH_ORDER)  # high order for BSM ArrayDecoder

    # 4. Setup HRTF (MagLS for ASM)
    if IS_MAGLS_HRTF:
        print("Computing MagLS HRTF for ASM...")
        hrtf = magls_hrtf(hrtf=hrtf_copy, sh_order=1, cutoff_over_freq=MAGLS_CUTOFF)
        # hrtf: (2, nm, T) — time + SH domain
    else:
        hrtf = hrtf_copy
        hrtf.toSH(1)

    # 5. Load and Add Source (Directly passing path)
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)
    print(f"Source added at {SOURCE_POS}, Receiver at {HEAD_POS}")

    # 6. Simulate Room (Ambisonics)
    print("Simulating Room (Ambisonics)...")
    amb_room = room.compute_amb()
    print(f"Room Ambisonics Shape: {amb_room.data.shape}")

    # 7. Process Chains
    asm_chain = ProcessorChain([
        ArrayDecoder(array_time_sh),        # Amb → mic signals (SH order 1)
        ASMEncoder(asm),                    # mic signals → Ambisonics (N=1)
        BinauralDecoder(hrtf),              # Ambisonics → binaural (MagLS HRTF)
    ])

    bsm_chain = ProcessorChain([
        ArrayDecoder(array_sh_bsm),         # Amb → mic signals (SH order 20)
        BSMEncoder(bsm),                    # mic signals → binaural (BSM+MagLS)
    ])

    print("Processing ASM chain...")
    asm_binaural = asm_chain.process(amb_room)

    print("Processing BSM chain...")
    bsm_binaural = bsm_chain.process(amb_room)

    # 8. Listen and Plot
    room.plot(extra_obj=hrtf)
    plt.show()

    print("Playing ASM binaural...")
    play_audio(asm_binaural, FS)

    print("Playing BSM + MagLS binaural...")
    play_audio(bsm_binaural, FS)


if __name__ == "__main__":
    main()