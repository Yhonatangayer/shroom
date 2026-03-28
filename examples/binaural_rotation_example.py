import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from shroom.acoustics.room import Room
from shroom.acoustics.processors import BinauralDecoder
from shroom.utils.file_utils import load_file
from devutils.sound import play_audio
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# --- Configuration ---
FS = 48000
SH_ORDER = 3
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.8
MAX_ISM_ORDER = 5
HEAD_POS = [2.0, 2.0, 1.5]
SOURCE_POS = [4.0, 2.0, 1.5]  # Source directly in front (X axis)


def main():
    print("--- Binaural Rotation Example (Rotating HRTF) ---")

    # 1. Initialize Room
    room = Room(
        dimensions=ROOM_DIMS,
        absorption=ABSORPTION,
        max_ism_order=MAX_ISM_ORDER,
        fs=FS,
        sh_order=SH_ORDER,
    )

    # 2. Load HRTF
    print(f"Loading HRTF from {DEFAULT_HRTF_PATH}...")
    hrtf_base = load_file(DEFAULT_HRTF_PATH)
    hrtf_base.resample(desired_fs=FS)
    hrtf_base.toSH(N_sp=SH_ORDER)

    # 3. Add Source
    print(f"Adding source from {DEFAULT_WAV_PATH}...")
    room.add_source(SOURCE_POS, signal=DEFAULT_WAV_PATH)
    room.set_receiver(HEAD_POS)

    # 4. Compute Ambisonics (Reference Frame)
    print("Computing Ambisonics Response...")
    amb_ref = room.compute_amb()

    # 5. Rotate and Decode
    # We will rotate the HRTF (Listener Orientation).
    # Rotations: (Yaw, Pitch, Roll)
    rotations = [
        (-45, 0, 0),
        (45, 0, 0),
        (0, 0, 0),
        (0, 30, 0),
        (0, 60, 0),
    ]

    for yaw, pitch, roll in rotations:
        print(
            f"\n--- Simulating Head Rotation: Yaw={yaw}, Pitch={pitch}, Roll={roll} ---"
        )

        # Copy original HRTF
        hrtf_rot = hrtf_base.copy()

        # Create rotation
        # Rotating the Head (HRTF).
        # Sequence: z (yaw), y (pitch), x (roll) usually.
        # Or z-y-z for Euler.
        # Here we use 'zyx' (intrinsic) which is common for head tracking.
        rot = Rotation.from_euler("zyx", [yaw, pitch, roll], degrees=True)

        # Apply SH rotation to HRTF
        hrtf_rot.rotate_sh_domain(rot)

        # Decode using rotated HRTF
        decoder = BinauralDecoder(hrtf_rot, sh_order=SH_ORDER, output_format="numpy.ndarray")
        binaural = decoder.process(amb_ref)

        # Plot Room with Rotated HRTF Orientation (XY and XZ)
        print("Plotting setup...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # XY Plane (Top View)
        room.plot(ax=ax1, extra_obj=hrtf_rot, plane="xy")
        ax1.set_title(f"Top View (XY) - Yaw: {yaw}")

        # XZ Plane (Side View)
        room.plot(ax=ax2, extra_obj=hrtf_rot, plane="xz")
        ax2.set_title(f"Side View (XZ) - Pitch: {pitch}")

        plt.tight_layout()
        plt.show(block=False)
        plt.pause(1.5)  # Show plot for 1.5 seconds

        # Play
        print(f"Playing...")
        play_audio(binaural, FS)

        plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
