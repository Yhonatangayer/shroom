"""
shroom — Spherical Harmonics ROOM
=================================
Python library for room acoustics simulation using Ambisonics,
with binaural rendering, spherical array simulation, and head rotation.
"""

from shroom.acoustics.room import Room
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.acoustics.processors import (
    BinauralDecoder,
    ArrayDecoder,
    ASMEncoder,
    BSMEncoder,
    ProcessorChain,
)
from shroom.acoustics.hrtf_processing import magls_hrtf
from shroom.acoustics.spherical_array import SphericalArray
from shroom.encoders.asm import ASM
from shroom.encoders.bsm import BSM
from shroom.utils.file_utils import load_file
from shroom.utils.rotation_utils import wigner_d_matrix
from shroom.paths import get_default_hrtf_path

from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("pyshroom")
except PackageNotFoundError:
    __version__ = "0.1.0"  # fallback for editable / uninstalled

__all__ = [
    "Room",
    "SpatialSignal",
    "BinauralDecoder",
    "ArrayDecoder",
    "ASMEncoder",
    "BSMEncoder",
    "ProcessorChain",
    "SphericalArray",
    "ASM",
    "BSM",
    "magls_hrtf",
    "load_file",
    "wigner_d_matrix",
    "get_default_hrtf_path",
    "__version__",
]
