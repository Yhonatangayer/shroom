import warnings

import numpy as np
from scipy.fft import fft, ifft, next_fast_len


def reconstruct_neg_frequency_spectrum(s: np.ndarray, n_fft: int, freq_axis: int = 0) -> np.ndarray:
    """
    Construct a Hermitian-symmetric full spectrum from the positive-frequency side.

    Parameters
    ----------
    s : ndarray
        Array containing the DC + positive frequency bins (rfft output).
    n_fft : int
        The number of samples in the original time-domain signal.
    freq_axis : int, optional
        Axis along which the concatenation is applied. Default is 0.

    Returns
    -------
    out : ndarray
        Hermitian-symmetric full FFT spectrum suitable for np.fft.ifft.
    """
    s = np.asarray(s)
    
    # Validate input shape
    n_pos_freqs = (n_fft // 2) + 1
    if s.shape[freq_axis] != n_pos_freqs:
        raise ValueError(
            f"Input size {s.shape[freq_axis]} on freq_axis does not match "
            f"the expected rfft output size {n_pos_freqs} for n_samples={n_fft}."
        )

    # Move target axis to front for simplicity
    s_T = np.moveaxis(s, freq_axis, 0)  # shape: (Fpos, ...)

    # Determine if the original signal length was even or odd
    is_even = n_fft % 2 == 0

    if is_even:
        # For even length, Nyquist is unique.
        # Negative frequencies are conjugate of positive frequencies (excluding DC and Nyquist).
        neg = s_T[1:-1].conj()[::-1]
    else:
        # For odd length, there is no unique Nyquist bin.
        # Negative frequencies are conjugate of all positive frequencies (excluding DC).
        neg = s_T[1:].conj()[::-1]

    # Concatenate: [DC, positives, (Nyquist if even), negative frequencies]
    out_T = np.concatenate([s_T, neg], axis=0)

    # Move back to original axis placement
    out = np.moveaxis(out_T, 0, freq_axis)
    return out


def convolve_and_sum(
    signal1: np.ndarray, signal2: np.ndarray, signal1_domain: str, signal2_domain: str
) -> np.ndarray:
    """
    Parameters:
    ----------
    signal1 : np.ndarray, shape (N1, ch, T1) - Ambisonics
    signal2 : np.ndarray, shape (N2, ch, T2) - HRTFs or RIRs
    signal1_domain: str, 'time' or 'freq' represents signal1 domain.
    signal2_domain: str, 'time' or 'freq' represents signal2 domain.

    Returns:
    -------
    output : np.ndarray, shape (N1, N2, T1 + T2 - 1)
    """
    # 1. Strict Domain Enforcement
    if signal1_domain != "time" or signal2_domain != "time":
        raise ValueError(
            f"Domain Mismatch: Both signals must be in 'time' domain. "
            f"Received: signal1={signal1_domain}, signal2={signal2_domain}. "
            f"Please convert signals to time domain before calling this function "
            f"to ensure correct linear convolution padding."
        )
    # 2. Input Validation
    assert signal1.ndim == 3, "signal1 must be 3D: (N1, ch, T1)"
    assert signal2.ndim == 3, "signal2 must be 3D: (N2, ch, T2)"

    N1, ch1, T1 = signal1.shape
    N2, ch2, T2 = signal2.shape

    if ch1 != ch2:
        raise ValueError(
            f"signal1 number of channels ({ch1}) must match signal2 number of channels ({ch2})."
        )

    L_out   = T1 + T2 - 1
    T_short = min(T1, T2)
    T_long  = max(T1, T2)

    # --- Choose strategy ---
    # OLA is beneficial when one signal is >> 8× longer than the other,
    # because it avoids padding the short filter to the full signal length.
    use_ola = (T_long > 8 * T_short) and (T_long > 1000)

    if not use_ola:
        # --- Full-FFT path (original algorithm) ---
        fft_len  = next_fast_len(L_out)
        S1       = fft(signal1[:, np.newaxis, :, :], n=fft_len, axis=-1)
        S2       = fft(signal2[np.newaxis, :, :, :], n=fft_len, axis=-1)
        S_result = np.sum(S1 * S2, axis=2)
        output   = ifft(S_result, n=fft_len, axis=-1)
        return output[..., :L_out]

    # --- OLA path ---
    # Block the longer signal; treat the shorter one as the "filter" whose
    # FFT is pre-computed once. Each block generates partial output that is
    # overlap-added into the result array.
    out_dtype = np.result_type(signal1.dtype, signal2.dtype, np.complex64)
    output    = np.zeros((N1, N2, L_out), dtype=out_dtype)

    block = next_fast_len(T_short * 8)
    step  = block - T_short + 1          # non-overlapping input step per block

    if T1 <= T2:
        # signal1 is the short "filter"; block-process signal2
        H_pre    = fft(signal1[:, np.newaxis, :, :], n=block, axis=-1, workers=-1)  # (N1, 1, ch, block)
        n_blocks = int(np.ceil(T2 / step))
        for k in range(n_blocks):
            start   = k * step
            end     = min(start + step, T2)
            chunk   = signal2[np.newaxis, :, :, start:end]              # (1, N2, ch, chunk_len)
            S_block = fft(chunk, n=block, axis=-1, workers=-1)          # (1, N2, ch, block)
            Y_block = np.sum(H_pre * S_block, axis=2)                   # (N1, N2, block)
            y_block = ifft(Y_block, n=block, axis=-1, workers=-1)       # (N1, N2, block)
            out_end = min(start + block, L_out)
            output[:, :, start:out_end] += y_block[:, :, :out_end - start]
    else:
        # signal2 is the short "filter"; block-process signal1
        H_pre    = fft(signal2[np.newaxis, :, :, :], n=block, axis=-1, workers=-1)  # (1, N2, ch, block)
        n_blocks = int(np.ceil(T1 / step))
        for k in range(n_blocks):
            start   = k * step
            end     = min(start + step, T1)
            chunk   = signal1[:, np.newaxis, :, start:end]              # (N1, 1, ch, chunk_len)
            S_block = fft(chunk, n=block, axis=-1, workers=-1)          # (N1, 1, ch, block)
            Y_block = np.sum(S_block * H_pre, axis=2)                   # (N1, N2, block)
            y_block = ifft(Y_block, n=block, axis=-1, workers=-1)       # (N1, N2, block)
            out_end = min(start + block, L_out)
            output[:, :, start:out_end] += y_block[:, :, :out_end - start]

    return output


def convolve_multichannel(signal: np.ndarray, filters: np.ndarray) -> np.ndarray:
    """
    Convolve a single signal with multiple filters (channels) efficiently.

    Uses Overlap-Add (OLA) with block size ≈ 8 × T_filt so each FFT window is
    much smaller than the full signal length.  Filter FFTs are pre-computed
    once and scipy.fft workers=-1 enables multi-threaded transforms.

    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array). Shape (T_sig,).
    filters : np.ndarray
        Filter bank. Shape (Channels, T_filt).

    Returns
    -------
    output : np.ndarray
        Convolved signals. Shape (Channels, T_sig + T_filt - 1).
    """
    signal  = np.asarray(signal)
    filters = np.asarray(filters)

    if signal.ndim != 1:
        raise ValueError("signal must be 1D array.")
    if filters.ndim != 2:
        raise ValueError("filters must be 2D array (Channels, Time).")

    T_sig              = signal.shape[0]
    n_channels, T_filt = filters.shape
    L_out              = T_sig + T_filt - 1

    # --- Choose OLA block size ---
    # block ≈ 8 × T_filt keeps FFT windows small when signal >> filter.
    # Falls back to a single block when the signal is short.
    block   = next_fast_len(max(T_filt * 8, L_out))
    step    = block - T_filt + 1        # non-overlapping input step

    if step <= 0 or step >= L_out:
        # Degenerate: use a single full-length FFT (original behaviour)
        fft_len = next_fast_len(L_out)
        S = fft(signal,  n=fft_len, workers=-1)
        H = fft(filters, n=fft_len, axis=-1, workers=-1)
        y = ifft(S[np.newaxis, :] * H, n=fft_len, axis=-1, workers=-1)
        return y[:, :L_out]

    # Pre-compute filter FFTs once at the chosen block size
    H_pre = fft(filters, n=block, axis=-1, workers=-1)   # (Channels, block)

    # ifft always returns complex; promote to the matching complex dtype
    out_dtype = np.result_type(filters.dtype, np.complex64)
    output    = np.zeros((n_channels, L_out), dtype=out_dtype)
    n_blocks = int(np.ceil(T_sig / step))

    for k in range(n_blocks):
        start   = k * step
        end     = min(start + step, T_sig)
        S_block = fft(signal[start:end], n=block, workers=-1)   # (block,)
        y_block = ifft(S_block[np.newaxis, :] * H_pre,
                       n=block, axis=-1, workers=-1)             # (Channels, block)
        out_end = min(start + block, L_out)
        output[:, start:out_end] += y_block[:, : out_end - start]

    return output


def reconstruct_frequency_sh_spectrum_full(H_pos, n_fft=None):
    """
    Reconstruct the full complex frequency spectrum for SH-domain signals.
    Handles both EVEN and ODD FFT lengths.

    Assumes input shape (..., nm, freq).

    Parameters
    ----------
    H_pos : ndarray
        One-sided spectrum (rfft output). Shape (..., nm, K_pos).
    n_fft : int, optional
        The length of the original time-domain signal (FFT size).
        If None, assumes Even length: 2 * (K_pos - 1).
        **Must be provided if the original length was Odd.**
    """
    # Assume last axis is freq, second to last is nm
    nm = H_pos.shape[-2]
    K_pos = H_pos.shape[-1]

    N = int(np.sqrt(nm) - 1)
    nm_list = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            nm_list.append((n, m))

    # 1. Determine Full FFT Size
    if n_fft is None:
        F = 2 * (K_pos - 1)  # Default to Even assumption
    else:
        F = n_fft

    # Validate input shape matches expected rfft size
    expected_K = (F // 2) + 1
    if K_pos != expected_K:
        raise ValueError(
            f"Input freq size {K_pos} does not match n_fft={F} (expected {expected_K})"
        )

    # Create output shape: (..., nm, F)
    out_shape = list(H_pos.shape)
    out_shape[-1] = F
    H_full = np.zeros(out_shape, dtype=complex)

    # Copy positive frequencies (0 to Nyquist-ish)
    H_full[..., :K_pos] = H_pos

    nm_to_index = {nm_list[i]: i for i in range(len(nm_list))}

    # 2. Define Indices based on Even/Odd
    if F % 2 == 0:
        # EVEN Case: Has Nyquist at index F/2
        idx_nyq = K_pos - 1
        k_pos = np.arange(1, idx_nyq)  # Exclude DC and Nyquist
        k_neg = F - k_pos  # Map to upper half
        has_nyquist = True
    else:
        # ODD Case: No Nyquist bin
        k_pos = np.arange(1, K_pos)  # Exclude DC only
        k_neg = F - k_pos
        has_nyquist = False

    for idx, (n, m) in enumerate(nm_list):
        if m == 0:
            # Standard Hermitian (Real in time)
            # H_full[..., idx, k_neg] = H_pos[..., idx, k_pos].conj()
            H_full[..., idx, k_neg] = H_pos[..., idx, k_pos].conj()

            if has_nyquist:
                H_full[..., idx, idx_nyq].imag = 0.0

        else:
            idx_minus = nm_to_index[(n, -m)]
            parity = (-1) ** m

            # Fill Negative Frequencies
            # H_full[..., idx_minus, k_neg] = parity * H_pos[..., idx, k_pos].conj()
            H_full[..., idx_minus, k_neg] = parity * H_pos[..., idx, k_pos].conj()

            # Enforce Nyquist Consistency (Only for Even)
            if has_nyquist:
                val_nyq = H_pos[..., idx, idx_nyq]
                H_full[..., idx_minus, idx_nyq] = parity * val_nyq.conj()

    return H_full


def is_signal_frequency_symmetric(
    signal: np.ndarray, freq_axis: int = -1, atol: float = 1e-5
) -> bool:
    """
    Check if the frequency spectrum is Hermitian symmetric (corresponds to real time signal).

    Parameters
    ----------
    signal : np.ndarray
        Frequency domain signal.
    freq_axis : int
        Axis corresponding to frequency.
    atol : float
        Absolute tolerance for comparison.

    Returns
    -------
    bool
        True if symmetric.
    """
    # Move freq axis to front
    s = np.moveaxis(signal, freq_axis, 0)
    n_bins = s.shape[0]

    # Check DC (index 0) is real
    if not np.allclose(s[0].imag, 0, atol=atol):
        warnings.warn(
            f"Symmetry Failed: DC component is not real. Max imag: {np.max(np.abs(s[0].imag))}"
        )
        return False

    # Check Nyquist (if even length)
    if n_bins % 2 == 0:
        # Usually FFT of real signal has N/2 + 1 bins for rfft, or N bins for fft.
        # If this is full FFT:
        # Nyquist is at index N/2
        nyq_idx = n_bins // 2
        # But wait, if it's full FFT, we check S[k] == S*[-k]

        # Positive frequencies: 1 to N/2 - 1
        # Negative frequencies: N-1 down to N/2 + 1

        # Let's assume standard full FFT layout
        # 0: DC
        # 1 .. N/2-1: Pos
        # N/2: Nyquist (if even)
        # N/2+1 .. N-1: Neg

        # Check Nyquist realness
        if not np.allclose(s[nyq_idx].imag, 0, atol=atol):
            warnings.warn("Symmetry Failed: Nyquist component is not real.")
            return False

        # Check symmetry
        pos = s[1:nyq_idx]
        neg = s[nyq_idx + 1 :][::-1]

        if not np.allclose(pos, neg.conj(), atol=atol):
            warnings.warn("Symmetry Failed: Positive and Negative frequencies do not match.")
            return False

    else:
        # Odd length
        # 0: DC
        # 1 .. (N-1)/2: Pos
        # (N+1)/2 .. N-1: Neg
        mid = (n_bins - 1) // 2
        pos = s[1 : mid + 1]
        neg = s[mid + 1 :][::-1]

        if not np.allclose(pos, neg.conj(), atol=atol):
            warnings.warn("Symmetry Failed: Positive and Negative frequencies do not match.")
            return False

    return True


def is_signal_frequency_space_valid(s, freq_axis=1):
    # check if signal for signal at f==0 is uniform
    dc = np.take(s, 0, axis=freq_axis)
    dc = np.abs(dc)
    if np.isclose(dc.mean(axis=0), dc.min(axis=0), atol=1e-6).all():
        freq0_ok = True
    else:
        freq0_ok = False
        warnings.warn("frequencyXspace domain signal is not constant across f=0!")

    if is_signal_frequency_symmetric(s, freq_axis=freq_axis):
        symmetric_ok = True
    else:
        symmetric_ok = False
    return symmetric_ok and freq0_ok


def is_signal_frequency_sh_valid(s, freq_axis=2, sh_axis=1, atol=10e-8, rtol=1e-7):
    """
    Check if a complex SH frequency spectrum satisfies the reality condition:
    A_{n,-m}[F-k] = (-1)^m * A_{n,m}[k]*.

    Parameters
    ----------
    s : ndarray, shape (nm, F, ...)
        The full complex SH frequency spectrum.
    freq_axis : int
        The axis corresponding to the full frequency dimension (F). Default is 1.
    sh_axis : int
        The axis corresponding to the SH channel dimension (nm). Default is 0.
    atol : float
        Absolute tolerance for np.allclose comparison.
    rtol : float
        Relative tolerance for np.allclose comparison.

    Returns
    -------
    ok : bool
        True if the spectrum satisfies the complex SH reality condition, False otherwise.
    """
    # Move axes: sh -> 0, freq -> 1
    s_canonical = np.moveaxis(s, [sh_axis, freq_axis], [0, 1])

    nm = s_canonical.shape[0]
    F = s_canonical.shape[1]

    # Calculate max degree N from number of SH channels (nm = (N+1)^2)
    N = int(np.sqrt(nm) - 1)

    # Generate the (n, m) list corresponding to the channel indices
    nm_list = []
    for n in range(N + 1):
        for m in range(-n, n + 1):
            nm_list.append((n, m))
    nm_to_index = {nm_list[i]: i for i in range(len(nm_list))}

    # Indices for interior positive frequencies (1 to F/2 - 1)
    k_pos = np.arange(1, F // 2)

    # Flags to track failure reasons
    ok_pairs = True
    ok_dc_nyq = True

    # --- 2. Check SH Symmetry for all m ---
    for idx, (n, m) in enumerate(nm_list):
        # We only need to check channels where m >= 0. The symmetry definition
        # (A_{n,-m} defined by A_{n,m}) automatically covers m < 0.
        if m < 0:
            continue

        # Extract positive frequencies (k=1 to F/2 - 1) for A_{n,m}
        A_pos = s_canonical[idx, k_pos, ...]

        if m == 0:
            # --- m=0 Case: Standard Hermitian Symmetry A_{n,0}[F-k] = A_{n,0}[k]* ---
            # Extract corresponding negative frequencies (F-k) for A_{n,0}
            A_neg = s_canonical[idx, F - k_pos, ...]
            A_pos_conj = A_pos.conj()

            # Check frequency pairs
            pair_ok = np.allclose(A_neg, A_pos_conj, atol=atol, rtol=rtol)
            if not pair_ok:
                warnings.warn(
                    f"SH Symmetry Failed: Channel (n={n}, m={m}) frequency pairs do not match standard Hermitian."
                )
                ok_pairs = False

            # Check DC (k=0) must be real
            dc = s_canonical[idx, 0, ...]
            dc_ok = np.allclose(dc.imag, 0, atol=atol)
            if not dc_ok:
                warnings.warn(
                    f"SH Symmetry Failed: Channel (n={n}, m={m}) DC component (k=0) is not real."
                )
                ok_dc_nyq = False

            # Check Nyquist (k=F/2) must be real (only if F is even)
            if F % 2 == 0:
                nyq = s_canonical[idx, F // 2, ...]
                nyq_ok = np.allclose(nyq.imag, 0, atol=atol)
                if not nyq_ok:
                    warnings.warn(
                        f"SH Symmetry Failed: Channel (n={n}, m={m}) Nyquist component (k=F/2) is not real."
                    )
                    ok_dc_nyq = False

        else:
            # --- m > 0 Case: Complex SH Symmetry A_{n,-m}[F-k] = (-1)^m * A_{n,m}[k]* ---

            idx_minus = nm_to_index[(n, -m)]

            # Extract negative frequencies (F-k) for the mirror channel A_{n,-m}
            A_neg_mirror = s_canonical[idx_minus, F - k_pos, ...]

            # Calculate the expected value: (-1)^m * A_{n,m}[k]*
            expected_val = ((-1) ** m) * A_pos.conj()

            # Check frequency pairs
            pair_ok = np.allclose(A_neg_mirror, expected_val, atol=atol, rtol=rtol)
            if not pair_ok:
                warnings.warn(
                    f"SH Symmetry Failed: Channel (n={n}, m={m}) and its mirror (-m={-m}) fail the complex SH symmetry."
                )
                ok_pairs = False

    # ---- check f=0 validity
    ok_f0 = True
    for idx, (n, m) in enumerate(nm_list):
        if m == 0:
            # For m=0, DC must be real
            if not np.allclose(s_canonical[idx, 0, ...].imag, 0, atol=atol):
                warnings.warn(f"DC Check Failed: Channel (n={n}, m=0) is not real.")
                ok_f0 = False
        elif m > 0:
            # For m > 0, the DC bin must satisfy: A_{n,-m}[0] = (-1)^m * A_{n,m}[0]*
            idx_minus = nm_to_index[(n, -m)]
            val_pos = s_canonical[idx, 0, ...]
            val_neg = s_canonical[idx_minus, 0, ...]
            expected = ((-1) ** m) * val_pos.conj()

            if not np.allclose(val_neg, expected, atol=atol, rtol=rtol):
                warnings.warn(f"DC Check Failed: Symmetry (n={n}, m={m}) at f=0.")
                ok_f0 = False

    # --- 3. Final Result ---
    ok = ok_pairs and ok_dc_nyq and ok_f0

    return ok


def is_sh_valid(Y: np.ndarray, sh_axis: int = 1, atol: float = 1e-5) -> bool:
    """
    Check if the Spherical Harmonics basis matrix is valid (symmetric properties).

    Parameters
    ----------
    Y : np.ndarray
        SH matrix.
    sh_axis : int
        Axis corresponding to SH coefficients.
    atol : float
        Absolute tolerance.

    Returns
    -------
    bool
        True if valid.
    """

    Y = np.moveaxis(Y, sh_axis, 0)
    n_sh = Y.shape[0]
    sh_order = int(np.sqrt(n_sh) - 1)

    if (sh_order + 1) ** 2 != n_sh:
        raise ValueError(f"Dimension {n_sh} is not a valid full SH order size.")

    for n in range(1, sh_order + 1):
        for m in range(1, n + 1):
            idx_pos = n**2 + n + m
            idx_neg = n**2 + n - m

            y_pos = Y[idx_pos]
            y_neg = Y[idx_neg]

            expected_neg = ((-1) ** m) * y_pos.conj()

            if not np.allclose(y_neg, expected_neg, atol=atol):
                warnings.warn(f"SH Basis Symmetry Failed: (n={n}, m={m}) vs (-m).")
                return False

    return True
