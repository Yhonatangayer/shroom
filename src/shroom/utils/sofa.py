import numpy as np
import sofar as sf

from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.geometry.sampling import sphereicalGrid


def is_time(sofa: sf.sofa.Sofa) -> bool:
    """Check if SOFA data is in Time Domain."""
    if hasattr(sofa, "Data_IR"):
        return True
    elif hasattr(sofa, "Data_Real"):
        return False
    else:
        raise ValueError("SOFA file does not contain Data_IR or Data_Real.")


def load_sofa(filepath: str) -> SpatialSignal:
    """Load SOFA file to SpatialSignal."""
    sofa = sf.read_sofa(filepath)
    fs = int(sofa.Data_SamplingRate)
    grid = parse_sofa_grid(sofa)
    data = parse_sofa_data(sofa)
    data = preprocess_sofa_data(data)
    return SpatialSignal(
        data=data, fs=fs, is_time=is_sofa_time(sofa), is_space=True, grid=grid
    )


def preprocess_sofa_data(sofa_data: np.ndarray, time_axis=-1):
    """Remove DC offset from SOFA data."""
    data = sofa_data - np.mean(sofa_data, axis=time_axis, keepdims=True)
    return data


def is_sofa_time(sofa: sf.sofa.Sofa) -> bool:
    """Check if SOFA is in time domain."""
    if hasattr(sofa, "Data_IR"):
        return True
    elif hasattr(sofa, "Data_Real"):
        return False
    else:
        raise ValueError("SOFA file does not contain Data_IR or Data_Real.")


def parse_sofa_grid(sofa: sf.sofa.Sofa) -> sphereicalGrid:
    """Parse SOFA grid into SamplingGrid."""
    az_rad, colat_rad = convert_sofa_to_radians(
        sofa.SourcePosition, sofa.SourcePosition_Type
    )
    weights = None
    if hasattr(sofa, "Data_SamplingWeight"):
        weights = sofa.Data_SamplingWeight.flatten()
    orientation = None
    if hasattr(sofa, "ListenerView"):
        assert len(sofa.ListenerView) == 1
        orientation = sofa.ListenerView[0]
    return sphereicalGrid(
        az=az_rad,
        co=colat_rad,
        weights=weights,
        orientation=orientation,
        sh_type="complex",
    )


def parse_sofa_data(sofa: sf.sofa.Sofa) -> np.ndarray:
    """Parse SOFA data into (Channels, Measurements, Time)."""
    if hasattr(sofa, "Data_IR"):
        var = "Data_IR"
        raw_data = sofa.Data_IR
    elif hasattr(sofa, "Data_Real"):
        var = "Data_Real"
        if hasattr(sofa, "Data_Imag"):
            raw_data = sofa.Data_Real + 1j * sofa.Data_Imag
        else:
            raw_data = sofa.Data_Real
    else:
        raise ValueError("SOFA file does not contain Data_IR or Data_Real.")

    try:
       dims = sofa.get_dimension(var)
    except Exception:
        dims = sofa._dimensions['Data_IR']
    
    if dims == ("M", "R", "N") or dims == "MRN":
        return np.transpose(raw_data, (1, 0, 2))
    elif dims == ("R", "M", "N") or dims == "RMN":
        return raw_data  # Already in your format
    else:
        # Handle rare cases or throw an error
        raise ValueError(f"Unexpected SOFA dimensions: {dims}")


def convert_sofa_to_radians(source_position, position_type):
    """Convert SOFA source positions to radians."""
    if position_type.lower() == "cartesian":
        x, y, z = source_position[:, 0], source_position[:, 1], source_position[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        az_deg = np.degrees(np.arctan2(y, x))
        el_deg = np.degrees(np.arcsin(z / r))
    else:
        az_deg = source_position[:, 0]
        el_deg = source_position[:, 1]
    az_rad = np.radians(az_deg)
    el_rad = np.radians(el_deg)
    colat_rad = (np.pi / 2) - el_rad
    az_rad = np.mod(az_rad, 2 * np.pi)
    return az_rad, colat_rad
