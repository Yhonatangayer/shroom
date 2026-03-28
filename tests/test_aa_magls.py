import pytest
import numpy as np
from shroom.acoustics.hrtf_processing import array_aware_magls_hrtf, magls_hrtf
from shroom.acoustics.spherical_array import SphericalArray
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.encoders.asm import ASM
from shroom.geometry.sampling import sphereicalGrid
from shroom.utils.grid_utils import from_fibonacci_grid
from shroom.utils.dsp_utils import is_signal_frequency_sh_valid


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def aa_setup():
    """Small, deterministic setup for AA-MagLS tests."""
    fs = 8000
    duration = 0.01
    n_fft = int(fs * duration)   # 80 samples
    Q = 50
    n_mics = 5
    sh_order = 1
    cutoff = 500.0

    mics_grid = sphereicalGrid(
        az=np.deg2rad(np.array([-90, -45, 0, 45, 90])),
        co=np.deg2rad(np.array([90, 90 + 18, 90 - 18, 90 + 18, 90])),
    )
    source_grid = from_fibonacci_grid(Q)

    array = SphericalArray(
        fs=fs,
        duration=duration,
        r_sphere=0.08,
        r_mics=np.full(n_mics, 0.08),
        source_grid=source_grid,
        mics_grid=mics_grid,
        sphere_type="rigid",
        sh_order_for_sm_calc=3,
        convert_to_time=False,
    )

    rng = np.random.default_rng(42)
    hrtf_time = rng.standard_normal((2, Q, n_fft))
    hrtf = SpatialSignal(
        data=hrtf_time, fs=fs, is_time=True, is_space=True, grid=source_grid
    )
    hrtf.toFreq()

    asm = ASM(sh_order=sh_order, array=array, fs=fs, duration=duration)

    n_pos = n_fft // 2 + 1
    cutoff_bin = int(np.argmin(np.abs(np.fft.rfftfreq(n_fft, 1 / fs) - cutoff)))

    return {
        "fs": fs,
        "n_fft": n_fft,
        "n_pos": n_pos,
        "Q": Q,
        "n_mics": n_mics,
        "sh_order": sh_order,
        "cutoff": cutoff,
        "cutoff_bin": cutoff_bin,
        "array": array,
        "hrtf": hrtf,
        "asm": asm,
        "source_grid": source_grid,
    }


# ---------------------------------------------------------------------------
# Output domain & shape
# ---------------------------------------------------------------------------

def test_aa_magls_output_is_sh_time(aa_setup):
    """Result must be in SH domain, time domain."""
    s = aa_setup
    result = array_aware_magls_hrtf(
        hrtf=s["hrtf"],
        asm=s["asm"],
        array=s["array"],
        sh_order=s["sh_order"],
        cutoff_over_freq=s["cutoff"],
    )
    assert result.is_sh, "output should be in SH domain"
    assert result.is_time, "output should be in time domain"


def test_aa_magls_output_shape(aa_setup):
    """Result shape is (2, nm, n_fft)."""
    s = aa_setup
    nm = (s["sh_order"] + 1) ** 2
    result = array_aware_magls_hrtf(
        hrtf=s["hrtf"],
        asm=s["asm"],
        array=s["array"],
        sh_order=s["sh_order"],
        cutoff_over_freq=s["cutoff"],
    )
    assert result.data.shape == (2, nm, s["n_fft"]), (
        f"Expected (2, {nm}, {s['n_fft']}), got {result.data.shape}"
    )


def test_aa_magls_output_sh_valid(aa_setup):
    """SH-domain result must satisfy Hermitian symmetry after FFT."""
    s = aa_setup
    result = array_aware_magls_hrtf(
        hrtf=s["hrtf"],
        asm=s["asm"],
        array=s["array"],
        sh_order=s["sh_order"],
        cutoff_over_freq=s["cutoff"],
    )
    result_f = result.copy()
    result_f.toFreq()
    assert is_signal_frequency_sh_valid(result_f.data, freq_axis=-1)


# ---------------------------------------------------------------------------
# Below-cutoff bins are unchanged (== LS solution)
# ---------------------------------------------------------------------------

def test_aa_magls_below_cutoff_equals_ls(aa_setup):
    """Positive-frequency bins strictly below cutoff_bin must equal the LS solution."""
    s = aa_setup
    cutoff_bin = s["cutoff_bin"]
    if cutoff_bin == 0:
        pytest.skip("cutoff_bin is 0, nothing to compare below cutoff")

    # LS reference: space+freq → SH+freq (positive bins)
    hrtf_ls = s["hrtf"].copy()
    hrtf_ls.toSH(s["sh_order"])
    ls_pos = hrtf_ls.data[..., : s["n_pos"]]  # (2, nm, n_pos)

    # AA-MagLS result in freq domain
    result = array_aware_magls_hrtf(
        hrtf=s["hrtf"],
        asm=s["asm"],
        array=s["array"],
        sh_order=s["sh_order"],
        cutoff_over_freq=s["cutoff"],
    )
    result.toFreq()
    aa_pos = result.data[..., : s["n_pos"]]  # (2, nm, n_pos)

    np.testing.assert_allclose(
        aa_pos[:, :, :cutoff_bin],
        ls_pos[:, :, :cutoff_bin],
        atol=1e-10,
        err_msg="AA-MagLS below-cutoff bins differ from LS solution",
    )


# ---------------------------------------------------------------------------
# Input validation errors
# ---------------------------------------------------------------------------

def test_aa_magls_validation_hrtf_not_space(aa_setup):
    """Raises ValueError when HRTF is in SH domain."""
    s = aa_setup
    hrtf_sh = s["hrtf"].copy()
    hrtf_sh.toSH(s["sh_order"])
    with pytest.raises(ValueError, match="space domain"):
        array_aware_magls_hrtf(hrtf_sh, s["asm"], s["array"], s["sh_order"], s["cutoff"])


def test_aa_magls_validation_array_not_freq(aa_setup):
    """Raises ValueError when array is in time domain."""
    s = aa_setup
    array_time = s["array"].copy()
    array_time.toTime()
    with pytest.raises(ValueError, match="frequency domain"):
        array_aware_magls_hrtf(s["hrtf"], s["asm"], array_time, s["sh_order"], s["cutoff"])


def test_aa_magls_validation_sh_order_zero(aa_setup):
    """Raises ValueError when sh_order < 1."""
    s = aa_setup
    with pytest.raises(ValueError, match="sh_order"):
        array_aware_magls_hrtf(s["hrtf"], s["asm"], s["array"], 0, s["cutoff"])


def test_aa_magls_validation_negative_cutoff(aa_setup):
    """Raises ValueError when cutoff_over_freq <= 0."""
    s = aa_setup
    with pytest.raises(ValueError, match="cutoff_over_freq"):
        array_aware_magls_hrtf(s["hrtf"], s["asm"], s["array"], s["sh_order"], -100.0)


def test_aa_magls_validation_grid_size_mismatch(aa_setup):
    """Raises ValueError when HRTF grid size doesn't match array grid size."""
    s = aa_setup
    # Build an HRTF with a different grid size
    rng = np.random.default_rng(99)
    wrong_Q = s["Q"] + 10
    wrong_source_grid = from_fibonacci_grid(wrong_Q)
    hrtf_wrong = SpatialSignal(
        data=rng.standard_normal((2, wrong_Q, s["n_fft"])),
        fs=s["fs"],
        is_time=True,
        is_space=True,
        grid=wrong_source_grid,
    )
    hrtf_wrong.toFreq()
    with pytest.raises(ValueError, match="[Ss]ource grid"):
        array_aware_magls_hrtf(
            hrtf_wrong, s["asm"], s["array"], s["sh_order"], s["cutoff"]
        )


def test_aa_magls_validation_sh_order_mismatch(aa_setup):
    """Raises ValueError when requested sh_order doesn't match asm.cnm."""
    s = aa_setup
    # asm was computed for sh_order=1; request sh_order=2 → nm mismatch
    with pytest.raises(ValueError, match="SH coefficients"):
        array_aware_magls_hrtf(s["hrtf"], s["asm"], s["array"], 2, s["cutoff"])


# ---------------------------------------------------------------------------
# Does not mutate input
# ---------------------------------------------------------------------------

def test_aa_magls_does_not_mutate_hrtf(aa_setup):
    """The input HRTF SpatialSignal must be unchanged after the call."""
    s = aa_setup
    hrtf_before = s["hrtf"].data.copy()
    array_aware_magls_hrtf(
        s["hrtf"], s["asm"], s["array"], s["sh_order"], s["cutoff"]
    )
    np.testing.assert_array_equal(s["hrtf"].data, hrtf_before)
