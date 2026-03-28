import numpy as np
try:
    from scipy.special import sph_harm_y
except ImportError:
    # Fallback for older SciPy versions where sph_harm exists but sph_harm_y does not
    try:
        from scipy.special import sph_harm
        def sph_harm_y(n, m, theta, phi):
            """
            Wrapper to make old sph_harm look like new sph_harm_y (hypothetically).
            Old: sph_harm(m, n, phi, theta)  (m=order, n=degree, phi=azimuth, theta=colatitude)
            New (assumed): sph_harm_y(n, m, theta, phi) (n=degree, m=order, theta=colatitude, phi=azimuth)
            """
            return sph_harm(m, n, phi, theta)
    except ImportError:
        # If neither exists (e.g. very new SciPy without sph_harm and we guessed wrong name),
        # we are in trouble. But let's assume one exists.
        raise ImportError("Could not import sph_harm or sph_harm_y from scipy.special")


def get_tilde_matrix(sh_order: int) -> np.ndarray:
    """
    Generates the Tilde matrix (T) for Spherical Harmonics.

    The Tilde matrix is used to handle the conjugation property of Spherical Harmonics
    when convolving two SH signals. Specifically, for complex SH:
    Y_nm* = (-1)^m Y_n,-m

    The matrix T maps the vector of SH coefficients to its "conjugate-like" permutation.
    T[idx(n,m), idx(n,-m)] = (-1)^m

    Parameters
    ----------
    sh_order : int
        Maximum SH order.

    Returns
    -------
    tilde : np.ndarray
        Matrix of shape ((N_sp+1)**2, (N_sp+1)**2).
    """
    n_coeffs = (sh_order + 1) ** 2
    tilde = np.zeros((n_coeffs, n_coeffs))

    def get_idx(n, m):
        return n**2 + n + m

    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            row = get_idx(n, m)
            col = get_idx(n, -m)  # Mirroring the order index
            tilde[row, col] = (-1) ** m

    return tilde


def sh_matrix(
    sh_order: int, az: np.ndarray, co: np.ndarray, sh_type: str = "complex"
) -> np.ndarray:
    """
    Compute Spherical Harmonics Matrix (Y).

    Parameters
    ----------
    sh_order : int
        Maximum SH order.
    az : np.ndarray
        Azimuth in radians.
    co : np.ndarray
        Colatitude in radians.
    sh_type : str, optional
        'real' or 'complex'. Default is 'complex'.

    Returns
    -------
    Y : np.ndarray
        SH matrix of shape (n_points, (N_sp+1)**2).
    """
    n_points = len(az)
    n_coeffs = (sh_order + 1) ** 2
    Y = np.zeros((n_points, n_coeffs), dtype=np.complex128)

    idx = 0
    for n in range(sh_order + 1):
        for m in range(-n, n + 1):
            # scipy.special.sph_harm(m, n, theta, phi)
            # theta = azimuth, phi = colatitude
            # New usage: sph_harm_y(n, m, co, az)
            Y[:, idx] = sph_harm_y(n, m, co, az)
            idx += 1

    if sh_type == "real":
        # Convert to Real SH using the standard ACN/N3D convention:
        #   m = 0 : Y_real = Re(Y_nm)
        #   m > 0 : Y_real = sqrt(2) * (-1)^m * Re(Y_nm)  [cosine-like]
        #   m < 0 : Y_real = sqrt(2) * (-1)^m * Im(Y_n|m|) [sine-like]
        Y_real = np.zeros((n_points, n_coeffs), dtype=np.float64)
        idx = 0
        for n in range(sh_order + 1):
            for m in range(-n, n + 1):
                if m == 0:
                    Y_real[:, idx] = np.real(Y[:, idx])
                elif m > 0:
                    Y_real[:, idx] = np.sqrt(2) * ((-1) ** m) * np.real(sph_harm_y(n, m, co, az))
                else:
                    Y_real[:, idx] = (
                        np.sqrt(2)
                        * ((-1) ** m)
                        * np.imag(sph_harm_y(n, abs(m), co, az))
                    )
                idx += 1
        return Y_real

    return Y
