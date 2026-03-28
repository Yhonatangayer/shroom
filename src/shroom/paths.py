import os

# Base directory of the package (src/shroom)
PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directory
DATA_DIR = os.path.join(PACKAGE_ROOT, "data")

# Default data file paths (may not exist on all installations)
DEFAULT_HRTF_PATH = os.path.join(DATA_DIR, "default_hrtf.sofa")
DEFAULT_WAV_PATH = os.path.join(DATA_DIR, "speech.wav")


def get_default_hrtf_path() -> str:
    """Return the default HRTF path, raising a clear error if the file is missing."""
    if not os.path.isfile(DEFAULT_HRTF_PATH):
        raise FileNotFoundError(
            f"Default HRTF not found at: {DEFAULT_HRTF_PATH}\n"
            "Please either:\n"
            "  1. Place a SOFA-format HRTF file at the path above, or\n"
            "  2. Pass an explicit hrtf_path to your pipeline.\n"
            "A suitable HRTF can be downloaded from:\n"
            "  https://zenodo.org/records/3928297 (TH Köln HRTF database)"
        )
    return DEFAULT_HRTF_PATH
