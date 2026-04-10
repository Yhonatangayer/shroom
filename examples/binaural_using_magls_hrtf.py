import matplotlib.pyplot as plt
from shroom.acoustics.room import Room
from shroom.acoustics.processors import BinauralDecoder
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.utils.file_utils import load_file
from shroom_dev.sound import play_audio
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# --- Configuration ---
FS = 48000.0
SH_ORDER = 1
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 5
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 4.0, 1.5]
HEAD_ROTATION = [ 0, 0, 0]


def main():
    print("--- Starting Binaural Room Simulation (MagLS HRTF) ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS, absorption=ABSORPTION, max_ism_order=MAX_ISM_ORDER, fs=FS
    )
    print(f"Room initialized: {ROOM_DIMS}m, Absorption: {ABSORPTION}")

    # 2. Load HRTF
    print(f"Loading HRTF from {DEFAULT_HRTF_PATH}...")
    hrtf = load_file(DEFAULT_HRTF_PATH)

    # 3. Compute MagLS HRTF (SH Domain)
    print(f"Computing MagLS HRTF (Order {SH_ORDER})...")
    hrtf_magls = magls_hrtf(hrtf, sh_order=SH_ORDER, cutoff_over_freq=1200)
    print(f"MagLS HRTF Shape: {hrtf_magls.data.shape}")

    # 4. Load and Add Source
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)

    # 5. Compute Ambisonics Response
    print(f"Computing Ambisonics Response (Order {SH_ORDER})...")
    amb_room = room.compute_amb()

    # 6. Decode to Binaural using MagLS HRTF
    print("Decoding to Binaural...")
    MagLSdecoder = BinauralDecoder(hrtf_magls, sh_order=SH_ORDER)
    binaural_output = MagLSdecoder.process(amb_room)

    # 7. Play Audio
    print("Playing Binaural Audio...")
    play_audio(binaural_output, FS)

    # Plot Room
    room.plot(extra_obj=hrtf_magls)
    plt.show()


if __name__ == "__main__":
    main()
