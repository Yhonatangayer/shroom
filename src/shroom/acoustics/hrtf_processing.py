from shroom.acoustics.spatial_signal import SpatialSignal
from typing import Optional
import numpy as np
from shroom.utils.math_utils import magls
from shroom.utils.dsp_utils import (
    reconstruct_frequency_sh_spectrum_full,
)
from shroom.utils.amb_utils import get_tilde_matrix


def _validate_magls_hrtf_inputs(hrtf, sh_order, cross_over_freq):
    if not hrtf.is_space:
        raise ValueError("magls_hrtf() expects input hrtf to be in space domain.")
    return

def _validate_aa_magls_hrtf_inputs(hrtf, asm, array, sh_order, cutoff_over_freq):
    if not hrtf.is_space:
        raise ValueError("array_aware_magls_hrtf() expects hrtf to be in space domain.")

    if not array.is_space:
        raise ValueError("array_aware_magls_hrtf() expects array to be in space domain.")

    if not array.is_freq:
        raise ValueError("array_aware_magls_hrtf() expects array to be in frequency domain.")

    if not asm.cnm.is_freq:
        raise ValueError("array_aware_magls_hrtf() expects asm.cnm to be in frequency domain.")

    if not asm.cnm.is_sh:
        raise ValueError("array_aware_magls_hrtf() expects asm.cnm to be in SH domain.")

    if not isinstance(sh_order, int) or sh_order < 1:
        raise ValueError(f"sh_order must be a positive integer, got {sh_order}.")

    if cutoff_over_freq <= 0:
        raise ValueError(f"cutoff_over_freq must be positive, got {cutoff_over_freq}.")

    if hrtf.fs is not None and array.fs is not None and hrtf.fs != array.fs:
        raise ValueError(
            f"Sampling rate mismatch: hrtf.fs={hrtf.fs} != array.fs={array.fs}."
        )

    expected_nm = (sh_order + 1) ** 2
    actual_nm = asm.cnm.data.shape[1]
    if actual_nm != expected_nm:
        raise ValueError(
            f"asm.cnm SH coefficients ({actual_nm}) do not match sh_order={sh_order} "
            f"(expected {expected_nm})."
        )

    array_mics = array.data.shape[0]
    cnm_mics = asm.cnm.data.shape[0]
    if array_mics != cnm_mics:
        raise ValueError(
            f"Mic count mismatch: array has {array_mics} mics, asm.cnm has {cnm_mics}."
        )

    array_sources = array.data.shape[1]
    hrtf_sources = hrtf.data.shape[1]
    if array_sources != hrtf_sources:
        raise ValueError(
            f"Source grid size mismatch: array has {array_sources} points, "
            f"hrtf has {hrtf_sources} points."
        )

    array_freqs = array.data.shape[2]
    cnm_freqs = asm.cnm.data.shape[2]
    if array_freqs != cnm_freqs:
        raise ValueError(
            f"FFT length mismatch: array has {array_freqs} bins, asm.cnm has {cnm_freqs}."
        )

def _create_double_ramp_alpha(n_bins, cutoff_bin, final_bin=None, factor=np.sqrt(2)):
    """
    Creates a frequency-dependent alpha parameter for MagLS blending.

    Parameters
    ----------
    n_bins : int
        Number of frequency bins.
    cutoff_bin : int
        The bin index where the transition starts.
    final_bin : int, optional
        The bin index where the transition ends (for band-pass behavior).
    factor : float, optional
        Transition width factor.

    Returns
    -------
    alpha : np.ndarray
        Array of weights in [0, 1].
    """
    idxs = np.arange(n_bins)

    p_up_start = cutoff_bin
    p_up_end = cutoff_bin * factor

    if final_bin is not None:
        p_down_end = final_bin

        # Symmetric descent: if the ramp-up ends at (cutoff * factor),
        # the ramp-down starts at (final / factor), guaranteeing start < end for factor > 1.
        p_down_start = final_bin / factor

        points = [
            (0, 0),
            (p_up_start, 0),
            (p_up_end, 1),
            (p_down_start, 1),
            (p_down_end, 0),
            (n_bins, 0),
        ]
    else:
        points = [(0, 0), (p_up_start, 0), (p_up_end, 1), (n_bins, 1)]

    # Sort by x-coordinate (required by np.interp)
    points.sort(key=lambda x: x[0])
    xp, fp = zip(*points)
    alpha = np.interp(idxs, xp, fp)

    return alpha


def magls_hrtf(
    hrtf: SpatialSignal,
    sh_order: Optional[int] = 1,
    cutoff_over_freq: Optional[float] = 1200,
):
    """
    Compute Magnitude Least Squares (MagLS) SH representation of an HRTF.

    This method blends standard Least Squares (LS) at low frequencies with
    Magnitude Least Squares (MagLS) at high frequencies. MagLS optimizes the
    SH coefficients to match the magnitude response, preserving timbre at the
    cost of phase accuracy.

    Parameters
    ----------
    hrtf : SpatialSignal
        Input HRTF in Space Domain.
    sh_order : int, optional
        Target SH order. Default is 1.
    cutoff_over_freq : float, optional
        Crossover frequency (Hz) between LS and MagLS. Default is 1200 Hz.

    Returns
    -------
    hrtf_magls : SpatialSignal
        Processed HRTF in SH Domain (Time).
    """

    hrtf_copy = hrtf.copy()

    _validate_magls_hrtf_inputs(hrtf_copy, sh_order, cutoff_over_freq)

    if hrtf_copy.is_time:
        hrtf_copy.toFreq()

    fs = hrtf_copy.fs
    duration = (1 / fs) * hrtf_copy.data.shape[2]

    freq_axis = np.fft.fftfreq(n=int(duration * fs), d=1 / fs)
    nfft = len(freq_axis)
    pos_freq_axis = np.fft.rfftfreq(n=int(duration * fs), d=1 / fs)
    pos_freqs_indices = freq_axis >= 0.0
    cutoff_bin = np.argmin(np.abs(pos_freq_axis - cutoff_over_freq))
    alpha = _create_double_ramp_alpha(len(pos_freq_axis), cutoff_bin)

    Y = hrtf_copy.grid.Y(sh_order)
    hrtf_space = hrtf_copy.data[..., : len(pos_freq_axis)]
    hrtf_copy.toSH(sh_order)
    hrtf_sh = hrtf_copy.data[..., : len(pos_freq_axis)]

    hnm_mag = hrtf_sh.copy()

    for f in range(cutoff_bin, len(pos_freq_axis) - 1):
        hnm_mag[0, :, f] = (
            alpha[f]
            * magls(
                A=Y,
                b=hrtf_space[0, :, f],
                x_prev=hnm_mag[0, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[0, :, f]
        )
        hnm_mag[1, :, f] = (
            alpha[f]
            * magls(
                A=Y,
                b=hrtf_space[1, :, f],
                x_prev=hnm_mag[1, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[1, :, f]
        )

    hnm_mag = reconstruct_frequency_sh_spectrum_full(hnm_mag, n_fft=nfft)

    hrtf_copy.data = hnm_mag
    hrtf_copy._log_change_to_history(
        "magls",
        {
            "sh_order": sh_order,
            "cutoff_over_freq": cutoff_over_freq,
            "space hrtf grid size": hrtf_space.shape[1],
        },
    )
    hrtf_copy.toTime()

    return hrtf_copy


def array_aware_magls_hrtf(
    hrtf: SpatialSignal,
    asm,
    array: SpatialSignal,
    sh_order: Optional[int] = 1,
    cutoff_over_freq: Optional[float] = 1200,
):
    """
    Compute Array-Aware Magnitude Least Squares (AA-MagLS) SH representation of an HRTF.

    Blends standard LS at low frequencies with MagLS at high frequencies, using
    A = V @ C_tilde^H as the system matrix instead of Y, where V is the array
    steering matrix and C_tilde is the tilde-permuted ASM filters. This incorporates
    knowledge of the microphone array into the HRTF optimization.

    Parameters
    ----------
    hrtf : SpatialSignal
        Input HRTF in Space Domain, sampled at the array's source grid.
    asm : ASM
        Ambisonics Signal Matching encoder with a `cnm` property returning a
        SpatialSignal of shape (n_mics, n_sh, n_freqs) in frequency domain.
    array : SpatialSignal
        Microphone array steering matrix in Frequency Domain.
        Shape: (n_mics, n_sources, n_freqs).
    sh_order : int, optional
        Target SH order. Default is 1.
    cutoff_over_freq : float, optional
        Crossover frequency (Hz) between LS and AA-MagLS. Default is 1200 Hz.

    Returns
    -------
    hrtf_aa_magls : SpatialSignal
        Processed HRTF in SH Domain (Time).
    """
    _validate_aa_magls_hrtf_inputs(hrtf, asm, array, sh_order, cutoff_over_freq)

    hrtf_copy = hrtf.copy()

    if hrtf_copy.is_time:
        hrtf_copy.toFreq()

    fs = hrtf_copy.fs
    duration = (1 / fs) * hrtf_copy.data.shape[2]

    freq_axis = np.fft.fftfreq(n=int(duration * fs), d=1 / fs)
    nfft = len(freq_axis)
    pos_freq_axis = np.fft.rfftfreq(n=int(duration * fs), d=1 / fs)
    n_pos = len(pos_freq_axis)

    cutoff_bin = np.argmin(np.abs(pos_freq_axis - cutoff_over_freq))
    alpha = _create_double_ramp_alpha(n_pos, cutoff_bin)

    # Space domain HRTF at positive frequencies: (2, Q, F_pos)
    hrtf_space = hrtf_copy.data[..., :n_pos]

    # LS solution in SH domain: (2, nm, F_pos)
    hrtf_copy.toSH(sh_order)
    hrtf_sh = hrtf_copy.data[..., :n_pos]

    # Build AA matrix A[q, n, f] = sum_m V[m,q,f] * (tilde @ C)[m,n,f]*
    # where V = steering matrix (M, Q, F), C = ASM filters (M, nm, F)
    tilde_matrix = get_tilde_matrix(sh_order)               # (nm, nm)
    sm_data = array.data[:, :, :n_pos]                      # (M, Q,  F_pos)
    cnm_data = asm.cnm.data[:, :, :n_pos]                   # (M, nm, F_pos)

    # C_tilde[m, n, f] = sum_k tilde[n, k] * cnm[m, k, f]
    C_tilde = np.einsum("nk,mkf->mnf", tilde_matrix, cnm_data)  # (M, nm, F_pos)

    # A[q, n, f] = sum_m V[m, q, f] * C_tilde[m, n, f]*
    A = np.einsum("mqf,mnf->qnf", sm_data, C_tilde.conj())      # (Q, nm, F_pos)

    hnm_mag = hrtf_sh.copy()

    for f in range(cutoff_bin, n_pos - 1):
        hnm_mag[0, :, f] = (
            alpha[f]
            * magls(
                A=A[:, :, f],
                b=hrtf_space[0, :, f],
                x_prev=hnm_mag[0, :, f - 1],
                A_prev=A[:, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[0, :, f]
        )
        hnm_mag[1, :, f] = (
            alpha[f]
            * magls(
                A=A[:, :, f],
                b=hrtf_space[1, :, f],
                x_prev=hnm_mag[1, :, f - 1],
                A_prev=A[:, :, f - 1],
            )
            + (1 - alpha[f]) * hrtf_sh[1, :, f]
        )

    hnm_mag = reconstruct_frequency_sh_spectrum_full(hnm_mag, n_fft=nfft)

    hrtf_copy.data = hnm_mag
    hrtf_copy._log_change_to_history(
        "aa_magls",
        {
            "sh_order": sh_order,
            "cutoff_over_freq": cutoff_over_freq,
            "space hrtf grid size": hrtf_space.shape[1],
            "array mics": sm_data.shape[0],
        },
    )
    hrtf_copy.toTime()

    return hrtf_copy