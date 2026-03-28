import numpy as np
from shroom.utils.amb_utils import get_tilde_matrix


def asm_mse_error(cnm, sm, Y, freqs, return_variance=False):
    """
    Calculates:
        |cnm^H V - Y^*|^2 / |Y|^2

    Parameters
    ----------
    cnm : np.ndarray, shape (M, nm, F)
    sm : np.ndarray, shape (M, Q, F)
    Y : np.ndarray, shape (Q, nm)
    freqs : np.ndarray
    return_variance : bool, optional
        If True, also return the variance band (std_q(|err_q|²) / mean_q(|Y_q|²))²
        per (nm, frequency). Default is False.

    Returns
    -------
    error : np.ndarray, shape (nm, F)
    variance : np.ndarray, shape (nm, F)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)  # (F, nm, M)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)    # (F, M, Q)
    Y = Y.T                                                  # (nm, Q)

    raw_err = cnm.conj() @ sm - Y[np.newaxis, ...].conj()  # (F, nm, Q)
    nominator = np.square(np.linalg.norm(raw_err, ord=2, axis=2))  # (F, nm)
    denominator = np.square(np.linalg.norm(Y, ord=2, axis=1))       # (nm,)
    error = nominator.T / denominator[..., np.newaxis]               # (nm, F)

    if return_variance:
        err_sq = np.abs(raw_err) ** 2                                    # (F, nm, Q)
        mean_pow = np.maximum(np.mean(np.abs(Y) ** 2, axis=1), 1e-12)  # (nm,)
        variance = (np.std(err_sq, axis=2) / mean_pow[np.newaxis, :]) ** 2  # (F, nm)
        variance = variance.T                                             # (nm, F)
        return error, variance
    return error


def asm_bin_mse_error(hnm, cnm, sm, h, freqs, return_variance=False):
    """
    Calculates:
        |tilde(hnm^T) cnm^H V - h^T|^2 / |h|^2

    Parameters
    ----------
    hnm : np.ndarray, shape (ears, nm, F)
    cnm : np.ndarray, shape (M, nm, F)
    sm : np.ndarray, shape (M, Q, F)
    h : np.ndarray, shape (ears, Q, F)
    freqs : np.ndarray
    return_variance : bool, optional
        If True, also return the variance band (std_q(|err_q|²) / mean_q(|h_q|²))²
        per (ear, frequency). Default is False.

    Returns
    -------
    error : np.ndarray, shape (ears, F)
    variance : np.ndarray, shape (ears, F)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)   # (F, nm, M)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)      # (F, M, Q)
    h = h[:, :, pos_freqs_indices].transpose(2, 0, 1)        # (F, ears, Q)
    hnm = hnm[:, :, pos_freqs_indices].transpose(2, 0, 1)    # (F, ears, nm)
    tilde = get_tilde_matrix(sh_order=int(np.sqrt(cnm.shape[1]) - 1))

    proj = cnm.conj() @ sm       # (F, nm, Q)
    res_tilde = tilde @ proj      # (F, nm, Q)
    final = hnm @ res_tilde       # (F, ears, Q)

    raw_err = final - h                                          # (F, ears, Q)
    nominator = np.square(np.linalg.norm(raw_err, axis=2))      # (F, ears)
    denominator = np.square(np.linalg.norm(h, axis=2))          # (F, ears)
    error = (nominator / denominator).T                          # (ears, F)

    if return_variance:
        err_sq = np.abs(raw_err) ** 2                                         # (F, ears, Q)
        mean_pow = np.maximum(np.mean(np.abs(h) ** 2, axis=2), 1e-12)        # (F, ears)
        variance = (np.std(err_sq, axis=2) / mean_pow) ** 2                   # (F, ears)
        variance = variance.T                                                   # (ears, F)
        return error, variance
    return error


def asm_bin_magnitude_mse_error(hnm, cnm, sm, h, freqs, return_variance=False):
    """
    Calculates:
        ||tilde(hnm^T) cnm^H V| - |h^T||^2 / |h|^2

    Parameters
    ----------
    hnm : np.ndarray, shape (ears, nm, F)
    cnm : np.ndarray, shape (M, nm, F)
    sm : np.ndarray, shape (M, Q, F)
    h : np.ndarray, shape (ears, Q, F)
    freqs : np.ndarray
    return_variance : bool, optional
        If True, also return the variance band (std_q(err_q²) / mean_q(|h_q|²))²
        per (ear, frequency). Default is False.

    Returns
    -------
    error : np.ndarray, shape (ears, F)
    variance : np.ndarray, shape (ears, F)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)   # (F, nm, M)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)      # (F, M, Q)
    h = h[:, :, pos_freqs_indices].transpose(2, 0, 1)        # (F, ears, Q)
    hnm = hnm[:, :, pos_freqs_indices].transpose(2, 0, 1)    # (F, ears, nm)
    tilde = get_tilde_matrix(sh_order=int(np.sqrt(cnm.shape[1]) - 1))

    proj = cnm.conj() @ sm       # (F, nm, Q)
    res_tilde = tilde @ proj      # (F, nm, Q)
    final = hnm @ res_tilde       # (F, ears, Q)

    raw_err = np.abs(final) - np.abs(h)                          # (F, ears, Q)
    nominator = np.square(np.linalg.norm(raw_err, axis=2))       # (F, ears)
    denominator = np.square(np.linalg.norm(h, axis=2))           # (F, ears)
    error = (nominator / denominator).T                           # (ears, F)

    if return_variance:
        err_sq = raw_err ** 2                                                  # (F, ears, Q)
        mean_pow = np.maximum(np.mean(np.abs(h) ** 2, axis=2), 1e-12)        # (F, ears)
        variance = (np.std(err_sq, axis=2) / mean_pow) ** 2                   # (F, ears)
        variance = variance.T                                                   # (ears, F)
        return error, variance
    return error


def linear_spectral_error(cnm, sm, Y, freqs, return_variance=False):
    """
    Calculates:
        |cnm^H V|^2 / |Y|^2

    Parameters
    ----------
    cnm : np.ndarray, shape (M, nm, F)
    sm : np.ndarray, shape (M, Q, F)
    Y : np.ndarray, shape (Q, nm)
    freqs : np.ndarray
    return_variance : bool, optional
        If True, also return the variance band (std_q(|proj_q|²) / mean_q(|Y_q|²))²
        per (nm, frequency). Default is False.

    Returns
    -------
    lse : np.ndarray, shape (nm, F)
    variance : np.ndarray, shape (nm, F)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    cnm = cnm[:, :, pos_freqs_indices].transpose(2, 1, 0)  # (F, nm, M)
    sm = sm[:, :, pos_freqs_indices].transpose(2, 0, 1)    # (F, M, Q)
    Y = Y.T                                                  # (nm, Q)

    raw_proj = cnm.conj() @ sm                                          # (F, nm, Q)
    nominator = np.square(np.linalg.norm(raw_proj, ord=2, axis=2))     # (F, nm)
    denominator = np.square(np.linalg.norm(Y, ord=2, axis=1))           # (nm,)
    lse = nominator.T / denominator[..., np.newaxis]                     # (nm, F)

    if return_variance:
        proj_sq = np.abs(raw_proj) ** 2                                      # (F, nm, Q)
        mean_pow = np.maximum(np.mean(np.abs(Y) ** 2, axis=1), 1e-12)      # (nm,)
        variance = (np.std(proj_sq, axis=2) / mean_pow[np.newaxis, :]) ** 2  # (F, nm)
        variance = variance.T                                                  # (nm, F)
        return lse, variance
    return lse
