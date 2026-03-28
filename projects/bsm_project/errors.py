import numpy as np


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
