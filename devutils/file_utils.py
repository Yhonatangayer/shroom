import pickle
from typing import Dict

import numpy as np
import yaml

from sh_room_sim.utils.file_utils import load_wav
from sh_room_sim.utils.sofa import load_sofa


def load_file_dev(file_path: str):
    """
    Load a file based on its extension.
    Supports .sofa, .wav, .yml, .yaml, .npz, .pkl.

    Parameters
    ----------
    file_path : str
        Path to the file.

    Returns
    -------
    Loaded object (SpatialSignal, tuple, dict, etc.)
    """
    suffix_to_format = {
        ".sofa": load_sofa,
        ".wav": load_wav,
        ".yml": load_yaml,
        ".yaml": load_yaml,
        ".npz": np.load,
        "pkl": load_pickle,
    }
    suffix = "." + file_path.split(".")[-1]
    if suffix in suffix_to_format:
        return suffix_to_format[suffix](file_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def load_pickle(pckl_path: str):
    """Load a pickle file."""
    return pickle.load(open(pckl_path, "rb"))


def load_yaml(yaml_path: str) -> Dict:
    """Load a YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data
