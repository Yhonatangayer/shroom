# shroom

**Spherical Harmonics Room**

[![CI](https://github.com/Yhonatangayer/shroom/actions/workflows/ci.yml/badge.svg)](https://github.com/Yhonatangayer/shroom/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

The package is published on PyPI as **`pyshroom`** (not `shroom`). There are two install flavors:

### 1. Minimal — just the core library

```bash
pip install pyshroom
```

Installs `shroom` and its runtime dependencies (numpy, scipy, matplotlib, pyroomacoustics, sofar). This is all you need to simulate rooms, encode Ambisonics, and render binaural audio from your own scripts.

### 2. With `shroom_dev` — extras for examples and benchmarks

```bash
pip install "pyshroom[dev]"
```

`shroom` is the library; **`shroom_dev` is an optional companion package** (installed via the
`[dev]` extra) holding evaluation metrics, plotting, and audio-playback helpers used by the
examples, tests, and benchmarks. **It is not required to use the core `shroom` library** —
only to run the examples/benchmarks in this repository. It bundles:

- `shroom_dev.plot` — `loglog_plot` for error curves with variance bands.
- `shroom_dev.sound` — `play_audio` helper around `sounddevice`.
- `shroom_dev.errors` — the ASM/BSM evaluation metrics used by the `benchmarks/` scripts (`asm_mse_error`, `asm_bin_mse_error`, `asm_bin_magnitude_mse_error`, `linear_spectral_error`, `bsm_mse_error`, `bsm_mag_mse_error`).
- `shroom_dev.file_utils` — extra file loaders.

The `[dev]` extra also pulls in `pytest`, `black`, `sounddevice`, and `pyyaml`.

### Running from a git checkout

If you cloned the repo to hack on the library itself, install it editable:

```bash
git clone https://github.com/Yhonatangayer/shroom.git
cd shroom
pip install -e ".[dev]"
```

You can then run the example scripts under `examples/` and the validation scripts under `benchmarks/` directly — they import from `shroom` and `shroom_dev`.

## _Quick Start_

### Basic Binaural Rendering

```python
import numpy as np
from shroom import Room
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
from shroom import Room, BinauralDecoder, load_file
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
from shroom import ProcessorChain, ArrayDecoder, ASMEncoder, BinauralDecoder, ASM

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
from shroom import magls_hrtf, BinauralDecoder

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

## Paper and Citation

If you use shroom in your research, please cite our paper:
[SHroom: A Python Framework for Ambisonics Room Acoustics Simulation and Binaural Rendering](https://arxiv.org/abs/2603.27342)
```bibtex
@misc{gayer2026shroompythonframeworkambisonics,
      title={SHroom: A Python Framework for Ambisonics Room Acoustics Simulation and Binaural Rendering}, 
      author={Yhonatan Gayer},
      year={2026},
      eprint={2603.27342},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2603.27342}, 
}
```
## Changelog

### 0.2.1

Maintenance release — no functional or API changes. Adds the JOSS paper
(`paper/`), continuous integration with coverage, contribution guidelines, and
expanded tests; renames the research `projects/` scripts to `benchmarks/` and
removes the unused `spaudiopy` submodule.

### 0.2.0

Three coupled changes. The first two are tied together (the pyroomacoustics upgrade
is what forced the absorption fix); the third is an independent array-model fix.

**1. pyroomacoustics ≥ 0.9 compatibility.** The minimum supported version was raised
from `0.7` to `0.9`. Newer pyroomacoustics removed the deprecated `absorption=` kwarg
in favour of `materials=pra.Material(...)`, which is what motivated the change below.

**2. BREAKING: `absorption` semantics fixed.** The `absorption` coefficient is now
applied directly as an **energy absorption** coefficient (e.g. `absorption=0.8` → 0.8),
matching `materials=pra.Material(0.8)`.

Previously the value was forwarded to pyroomacoustics' deprecated `absorption=` kwarg,
which silently converted it as `1 - (1 - a)**2` (so `0.8` became an effective `0.96`)
and emitted a `DeprecationWarning`. Simulated reverberation will therefore differ from
shroom < 0.2.0.

To reproduce results generated with an earlier version, pass `absorption_mode="legacy"`:

```python
Room(dimensions=[6.0, 5.0, 3.0], absorption=0.8)                       # 0.80 (new default)
Room(dimensions=[6.0, 5.0, 3.0], absorption=0.8, absorption_mode="legacy")  # 0.96 (pre-0.2.0)
```

**3. Spherical-array radial-function order mask removed (ghost-image fix).** The
per-order sigmoid mask `1 / (1 + exp(n − (ka+1)))` was removed from the radial
functions (`shroom.acoustics.physics`). Stacked across orders it formed a staircase of
sharp spectral edges that rings in the time domain and is heard as a duplicate/ghost
image for long radial filters (large `duration` ⇒ fine frequency resolution ⇒ ringing
past the ~10 ms echo-fusion window). Damping is now only the smooth Wiener-style
magnitude knee `|b_n|² / (|b_n|² + limit²)`, which already suppressed the same
numerically-insignificant coefficients without any order/frequency gate. The steering
matrix — and therefore ASM and AA-MagLS encoder filters — differ from shroom < 0.2.0;
the change removes the ghost image and moves the simulated array closer to true sphere
physics. No API changes.

## Contributing & Support

Contributions, bug reports, and questions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for development setup and guidelines. To report a bug, request a feature, or ask a question,
open an issue at <https://github.com/Yhonatangayer/shroom/issues>.

The `benchmarks/` directory contains validation scripts that reproduce the encoder-convergence
figures from the paper; see [benchmarks/README.md](benchmarks/README.md).

## License

MIT License
