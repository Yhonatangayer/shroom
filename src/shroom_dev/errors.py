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


def bsm_mse_error(cl, cr, array, hrtf, freqs, return_variance=False):
    """
    Calculates the BSM MSE error:
        mean_q |V_f @ c_f - h_f|^2 / mean_q |h_f|^2

    Variance (confidence band):
        (std_q(|err_q|^2) / mean_q(|h_q|^2))^2

    Parameters
    ----------
    cl : np.ndarray, shape (F, M)
    cr : np.ndarray, shape (F, M)
    array : SpatialSignal, shape (M, Q, F)
    hrtf : SpatialSignal, shape (2, Q, F)
    freqs : np.ndarray
    return_variance : bool, optional

    Returns
    -------
    mse_l, mse_r : np.ndarray, shape (F_pos,)
    var_l, var_r : np.ndarray, shape (F_pos,)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    V = array.data[:, :, pos_freqs_indices]   # (M, Q, F_pos)
    h = hrtf.data[:, :, pos_freqs_indices]    # (2, Q, F_pos)
    cl_pos = cl[pos_freqs_indices]             # (F_pos, M)
    cr_pos = cr[pos_freqs_indices]             # (F_pos, M)

    F_pos = V.shape[2]
    mse_l = np.zeros(F_pos)
    mse_r = np.zeros(F_pos)
    if return_variance:
        var_l = np.zeros(F_pos)
        var_r = np.zeros(F_pos)

    for f in range(F_pos):
        pred_l = V[:, :, f].T @ cl_pos[f, :].conj()  # (Q,)
        pred_r = V[:, :, f].T @ cr_pos[f, :].conj()  # (Q,)
        h_l = h[0, :, f]                              # (Q,)
        h_r = h[1, :, f]                              # (Q,)

        err_l = pred_l - h_l
        err_r = pred_r - h_r

        mse_l[f] = np.square(np.linalg.norm(err_l) / np.linalg.norm(h_l))
        mse_r[f] = np.square(np.linalg.norm(err_r) / np.linalg.norm(h_r))

        if return_variance:
            # Variance band: (std_q(|err_q|²) / mean_q(|h_q|²))²
            # Stored squared so that loglog_plot (which applies sqrt) recovers
            # the std-normalised quantity for the shaded confidence interval.
            mean_pow_l = np.maximum(np.mean(np.abs(h_l) ** 2), 1e-12)
            mean_pow_r = np.maximum(np.mean(np.abs(h_r) ** 2), 1e-12)
            var_l[f] = (np.std(np.abs(err_l) ** 2) / mean_pow_l) ** 2
            var_r[f] = (np.std(np.abs(err_r) ** 2) / mean_pow_r) ** 2

    if return_variance:
        return mse_l, mse_r, var_l, var_r
    return mse_l, mse_r


def bsm_mag_mse_error(cl, cr, array, hrtf, freqs, return_variance=False):
    """
    Calculates the BSM magnitude MSE error:
        mean_q (|V_f @ c_f| - |h_f|)^2 / mean_q |h_f|^2

    Variance (confidence band):
        (std_q((|pred_q| - |h_q|)^2) / mean_q(|h_q|^2))^2

    Parameters
    ----------
    cl : np.ndarray, shape (F, M)
    cr : np.ndarray, shape (F, M)
    array : SpatialSignal, shape (M, Q, F)
    hrtf : SpatialSignal, shape (2, Q, F)
    freqs : np.ndarray
    return_variance : bool, optional

    Returns
    -------
    mse_l, mse_r : np.ndarray, shape (F_pos,)
    var_l, var_r : np.ndarray, shape (F_pos,)  — only when return_variance=True
    """
    pos_freqs_indices = np.arange(len(freqs)) <= (len(freqs) // 2)

    V = array.data[:, :, pos_freqs_indices]   # (M, Q, F_pos)
    h = hrtf.data[:, :, pos_freqs_indices]    # (2, Q, F_pos)
    cl_pos = cl[pos_freqs_indices]             # (F_pos, M)
    cr_pos = cr[pos_freqs_indices]             # (F_pos, M)

    F_pos = V.shape[2]
    mse_l = np.zeros(F_pos)
    mse_r = np.zeros(F_pos)
    if return_variance:
        var_l = np.zeros(F_pos)
        var_r = np.zeros(F_pos)

    for f in range(F_pos):
        pred_l = np.abs(V[:, :, f].T @ cl_pos[f, :].conj())  # (Q,)
        pred_r = np.abs(V[:, :, f].T @ cr_pos[f, :].conj())  # (Q,)
        h_l_mag = np.abs(h[0, :, f])                          # (Q,)
        h_r_mag = np.abs(h[1, :, f])                          # (Q,)

        err_l = pred_l - h_l_mag
        err_r = pred_r - h_r_mag

        mse_l[f] = np.square(np.linalg.norm(err_l) / np.linalg.norm(h[0, :, f]))
        mse_r[f] = np.square(np.linalg.norm(err_r) / np.linalg.norm(h[1, :, f]))

        if return_variance:
            # Variance band: (std_q(err_q²) / mean_q(|h_q|²))²
            # Stored squared so that loglog_plot (which applies sqrt) recovers
            # the std-normalised quantity for the shaded confidence interval.
            mean_pow_l = np.maximum(np.mean(h_l_mag ** 2), 1e-12)
            mean_pow_r = np.maximum(np.mean(h_r_mag ** 2), 1e-12)
            var_l[f] = (np.std(err_l ** 2) / mean_pow_l) ** 2
            var_r[f] = (np.std(err_r ** 2) / mean_pow_r) ** 2

    if return_variance:
        return mse_l, mse_r, var_l, var_r
    return mse_l, mse_r
