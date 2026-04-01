# shroom

**Spherical Harmonics Room**

A Python library for simulating room acoustics using Spherical Harmonics (Ambisonics). It provides tools for simulating room impulse responses (ARIR), microphone arrays, and binaural rendering.

## Features

*   **Room Simulation**: Image Source Method (ISM) adapted for Spherical Harmonics.
*   **Spatial Signals**: Unified handling of Time, Frequency, Space, and Spherical Harmonics (SH) domains.
*   **Processors**: Modular processing chain including:
    *   `ArrayDecoder`: Simulates spherical microphone arrays.
    *   `ASMEncoder`: Encodes microphone signals to Ambisonics (ASM).
    *   `BinauralDecoder`: Decodes Ambisonics to Binaural audio using HRTFs.
*   **Rotation**: Efficient rotation of sound fields and HRTFs using Wigner-D matrices, or via space domain grid rotation.
*   **Visualization**: 2D and 3D plotting of room geometry, sources, and receiver orientation.

## Installation
**install pyshroom** and not shroom
```bash
pip install pyshroom
```

For development (including tests):

```bash
pip install -e .[dev]
```

## _Quick Start_

### Basic Binaural Rendering

```python
import numpy as np
from shroom.acoustics.room import Room
from shroom.paths import DEFAULT_WAV_PATH

# 1. Initialize Room
room = Room(
    dimensions=[6.0, 5.0, 3.0],
    absorption=0.8,
    sh_order=3,
    fs=48000
)

# 2. Add Source and Receiver
room.add_source([4.0, 2.0, 1.5], signal=DEFAULT_WAV_PATH)
room.set_receiver([2.0, 2.0, 1.5])

# 3. Compute Ambisonics Response
amb_signal = room.compute_amb()

# 4. Plot
room.plot(plot_3d=True)
```

### Dynamic Head Rotation

```python
import numpy as np
from scipy.spatial.transform import Rotation
from shroom.acoustics.room import Room
from shroom.acoustics.processors import BinauralDecoder
from shroom.utils.file_utils import load_file
from shroom.paths import DEFAULT_HRTF_PATH, DEFAULT_WAV_PATH

# 1. Initialize Room & Compute Ambisonics (Reference Frame)
room = Room(dimensions=[6.0, 5.0, 3.0], sh_order=3, fs=48000)
room.add_source([4.0, 2.0, 1.5], signal=DEFAULT_WAV_PATH)
room.set_receiver([2.0, 2.0, 1.5])
amb_ref = room.compute_amb()

# 2. Load HRTF
hrtf_base = load_file(DEFAULT_HRTF_PATH)
hrtf_base.toSH(N_sp=3)

# 3. Rotate Listener Orientation (Modal Rotation via Wigner-D)
rot = Rotation.from_euler("zyx", [45, 0, 0], degrees=True)  # 45 deg Yaw
hrtf_rot = hrtf_base.copy()
hrtf_rot.rotate_sh_domain(rot)

# 4. Decode with Rotated HRTF
decoder = BinauralDecoder(hrtf_rot, sh_order=3)
binaural = decoder.process(amb_ref)
```

### Complete ASM Processing Chain

```python
from shroom.acoustics.processors import ProcessorChain, ArrayDecoder, ASMEncoder, BinauralDecoder
from shroom.encoders.asm import ASM

# 1. Setup Signal Chain: Room -> Array -> ASM Encoder -> Binaural Decoder
# Note: array_time_sh and asm_instance must be pre-configured
chain = ProcessorChain([
    ArrayDecoder(array_time_sh),  # Simulate mic recordings
    ASMEncoder(asm_instance),  # Encode mics to Ambisonics (ASM)
    BinauralDecoder(hrtf, sh_order=1)  # Render to binaural
])

# 2. Process Ambisonics through the Chain
binaural_output = chain.process(room.compute_amb())
```

### Optimized Low-Order Rendering (MagLS)

```python
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.acoustics.processors import BinauralDecoder

# 1. Compute MagLS-optimized HRTF (Mitigates spectral artifacts at low SH orders)
hrtf_magls = magls_hrtf(original_hrtf, sh_order=1)

# 2. Decode using optimized modal weights
decoder = BinauralDecoder(hrtf_magls, sh_order=1)
binaural_output = decoder.process(room.compute_amb())
```

## Dependencies

*   numpy
*   scipy
*   matplotlib
*   pyroomacoustics
*   soundfile
*   sounddevice
*   sofar

## License

MIT License
