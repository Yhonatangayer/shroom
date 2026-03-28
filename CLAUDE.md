# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**shroom** (Modal Acoustics Spherical Harmonics ROOM) is a Python library for simulating room acoustics using Spherical Harmonics (Ambisonics). It supports binaural rendering via HRTFs, microphone array simulation, and real-time head rotation.

## Commands

```bash
# Install in development mode
pip install -e .[dev]

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_room.py

# Run a single test function
pytest tests/test_room.py::test_compute_amb -v

# Format code
black src/shroom tests/
```

## Architecture

### Central Data Structure: `SpatialSignal`

`src/shroom/acoustics/spatial_signal.py` is the core abstraction. All signals are represented with two independent domain axes:
- **Time ↔ Frequency** (`is_time` / `is_freq`)
- **Space ↔ Spherical Harmonics** (`is_space` / `is_sh`)

Data shape is always `(n_channels, n_grid_or_sh_coeffs, n_samples_or_freqs)`. Domain conversion methods (`toTime()`, `toFreq()`, `toSH()`, etc.) handle FFT and SH transforms in-place.

### Room Simulation Pipeline

```
Room (ISM via pyroomacoustics)
  → compute_arir()   → SpatialSignal [SH, Time]   (per-source ARIR)
  → compute_amb()    → SpatialSignal [SH, Time]   (mixed Ambisonics)
```

`Room` in `src/shroom/acoustics/room.py` wraps pyroomacoustics for image source computation, then applies SH decomposition and fractional delays internally. Supports per-wall `Material` objects or a global absorption coefficient.

### Processor Chain Pattern

`src/shroom/acoustics/processors.py` implements composable processors:
- `BinauralDecoder` — SH domain → stereo binaural (uses HRTF)
- `ArrayDecoder` — SH domain → microphone signals
- `ASMEncoder` — microphone signals → SH domain (wraps `ASM`)
- `ProcessorChain` — composes processors, computing a unified kernel for efficiency

Each processor implements `process(data: SpatialSignal) → SpatialSignal`.

### ASM Encoder

`src/shroom/encoders/asm.py` implements Ambisonics Signal Matching: designs frequency-domain filters `W = Y^H V^H (V V^H + εI)^-1` using Tikhonov regularization to encode mic array signals to Ambisonics.

### Spherical Array Simulation

`SphericalArray` in `src/shroom/acoustics/spherical_array.py` inherits from `SpatialSignal`. It computes the frequency-dependent steering matrix (transfer function between source grid and mic positions) for rigid or open sphere arrays using radial functions (Bn).

### Key Utilities

| Module | Purpose |
|--------|---------|
| `utils/dsp_utils.py` | FFT wrappers, convolution, Hermitian symmetry reconstruction |
| `utils/amb_utils.py` | SH matrix computation, ACN indexing, decoding matrices |
| `utils/rotation_utils.py` | Wigner-D matrices for SH rotation |
| `utils/math_utils.py` | Tikhonov regularization, MagLS optimization |
| `geometry/sampling.py` | `sphereicalGrid` — spherical sampling points with SH matrix and quadrature weights |

### HRTF Processing

`src/shroom/acoustics/hrtf_processing.py` provides MagLS optimization (`magls_hrtf()`) to reduce spectral artifacts when rendering at low SH orders by blending standard LS at low frequencies with magnitude-only LS at high frequencies.

### Data Flow Example

```
Room.compute_amb()          → SpatialSignal [SH, Time]
  ↓ BinauralDecoder.process()
  ↓  internally: toFreq() → convolve_sh(hrtf) → toTime()
Stereo SpatialSignal [Space, Time]  (2 channels: L/R)
```

## Testing Notes

- `tests/` contains reference `.npz` files (`anm_ref.npz`, `hnm_ref.npz`) used for numerical validation.
- Tests use `pytest` fixtures for room/array setup and compare outputs against reference data shapes, energy, and SH validity (zero DC for n>0 channels).
- `scipy` requires `>=1.10.0` (no upper bound).
