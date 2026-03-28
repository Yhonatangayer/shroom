import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from shroom.acoustics.room import Room
from shroom.acoustics.processors import BinauralDecoder
from shroom.utils.file_utils import load_file
from devutils.sound import play_audio
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# --- Configuration ---

FS = 48000.0
SH_ORDER = 14
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 10
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [ 90, 0, 0]


def main():
    print("--- Starting Binaural Room Simulation ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS, absorption=ABSORPTION, max_ism_order=MAX_ISM_ORDER, fs=FS
    )
    print(f"Room initialized: {ROOM_DIMS}m, Absorption: {ABSORPTION}")

    # 2. Load and Setup HRTF
    print(f"Loading HRTF from {DEFAULT_HRTF_PATH}...")
    hrtf = load_file(DEFAULT_HRTF_PATH)
    hrtf.resample(desired_fs=FS)

    # Rotate HRTF (Head Rotation)
    if HEAD_ROTATION is not None:
        print(f"Rotating Head by {HEAD_ROTATION} degrees (Yaw, Pitch, Roll)...")
        rot = Rotation.from_euler("zxy", HEAD_ROTATION, degrees=True)
        hrtf.rotate_space_domain(rot)

    hrtf.toSH(N_sp=SH_ORDER)

    # 3. Load and Add Source (Directly passing path)
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)
    print(f"Source added at {SOURCE_POS}, Receiver at {HEAD_POS}")

    # 4. Compute Ambisonics Response
    print(f"Computing Ambisonics Response (Order {SH_ORDER})...")
    amb_room = room.compute_amb()
    print(f"Ambisonics Shape: {amb_room.data.shape}")

    # 5. Decode to Binaural
    print("Decoding to Binaural...")
    decoder = BinauralDecoder(hrtf, sh_order=SH_ORDER)
    binaural_output = decoder.process(amb_room)
    print(f"Binaural Output Shape: {binaural_output.data.shape}")

    # 6. Play Audio
    print("Playing Binaural Audio...")
    # Output is (Channels, 1, Time) -> (2, 1, N)
    play_audio(binaural_output, FS)

    # Plot Room with HRTF Orientation
    room.plot(extra_obj=hrtf)
    plt.show()


if __name__ == "__main__":
    main()
