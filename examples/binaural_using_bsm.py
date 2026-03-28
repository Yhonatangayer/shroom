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
from shroom.encoders.bsm import BSM
from shroom.utils.file_utils import load_file
from devutils.sound import play_audio
from scipy.spatial.transform import Rotation
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# --- Configuration ---
FS = 48000.0
DURATION = 0.008
SM_SH_ORDER = 20          # accuracy of steering matrix computation
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 10
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [ 0, 0, 0]


def main():
    print("--- Starting Binaural Room Simulation (ASM Chain) ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS, absorption=ABSORPTION, max_ism_order=MAX_ISM_ORDER, fs=FS
    )
    print(f"Room initialized: {ROOM_DIMS}m, Absorption: {ABSORPTION}")

    # 2. Setup Array
    print(f"Generating Spherical Array...")
    mics_grid = sphereicalGrid(
        az=np.deg2rad(np.array([-90, -45, 0, 45, 90])),
        co=np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90])),
    )
    source_grid = from_fibonacci_grid(480)

    array = SphericalArray(
        source_grid=source_grid,
        mics_grid=mics_grid,
        r_mics=np.full(mics_grid.n_points, 0.08),
        fs=FS,
        duration=DURATION,
        r_sphere=0.08,
        sh_order_for_sm_calc=SM_SH_ORDER,
        convert_to_time=False,  # keep in frequency domain for ASM + AA-MagLS
    )
    # array = load_file("/daxta/sofa_hrtfs/qu_kemar_anechoic_2m.sofa")

    # Prepare array for ASM (Frequency Domain)
    array.resample(desired_fs=FS)
    array_freq = array.copy()
    array_freq.toFreq()
    array.toSH(SM_SH_ORDER)


    # 4. Setup HRTF
    print("Loading HRTF...")
    hrtf = load_file(DEFAULT_HRTF_PATH)

    hrtf.resample(desired_fs=FS)

    if HEAD_ROTATION is not None and not np.allclose(HEAD_ROTATION, [0, 0, 0]):
        print(f"Rotating Head by {HEAD_ROTATION} degrees (Yaw, Pitch, Roll)...")
        rot = Rotation.from_euler('zxy', HEAD_ROTATION, degrees=True)
        hrtf.rotate_space_domain(rot)

    # 3. Setup BSM
    hrtf.toSH(30)
    hrtf.toSpace(array_freq.grid)

    hrtf.zero_pad(int(DURATION*FS))
    hrtf.toFreq()
    bsm = BSM(
        array=array_freq,
        hrtf=hrtf,
        fs=FS,
        duration=DURATION,
        use_magls=True,
        magls_cutoff_frequency=1200.0
    )

    # 5. Load and Add Source (Directly passing path)
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)
    print(f"Source added at {SOURCE_POS}, Receiver at {HEAD_POS}")

    # 6. Simulate Room (Ambisonics)
    print("Simulating Room (Ambisonics)...")
    amb_room = room.compute_amb()
    print(f"Room Ambisonics Shape: {amb_room.data.shape}")

    # 7. Process Chain
    # Chain: Room(Amb) -> Array(Mics) -> BSM(binaural)
    chain = ProcessorChain(
        [
            ArrayDecoder(array),  # Simulate array recording
            BSMEncoder(bsm),  # Encode mics to binaurals
        ]
    )

    print("Processing Chain...")
    binaural_output = chain.process(amb_room)

    # 8. Listen and Plot
    print("Playing Binaural Audio...")
    play_audio(binaural_output, FS)

    # Plot Room with Array Orientation
    room.plot(extra_obj=hrtf)
    plt.show()

    print("yes!")


if __name__ == "__main__":
    main()