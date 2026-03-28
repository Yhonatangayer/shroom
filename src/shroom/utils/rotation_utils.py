import numpy as np
from scipy.special import factorial as fact

def wigner_d_matrix(N: int, alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the Wigner-D matrix for Spherical Harmonics rotation.

    The matrix D rotates SH coefficients such that:
    f_rot(omega) = f(R^-1 omega)
    c_rot = D(R) @ c

    Parameters
    ----------
    N : int
        Maximum SH order.
    alpha, beta, gamma : float
        Euler angles in radians (Z-Y-Z convention).
        Rotation R = Rz(alpha) * Ry(beta) * Rz(gamma).

    Returns
    -------
    D : np.ndarray
        Wigner-D matrix of shape ((N+1)^2, (N+1)^2).
        Block diagonal structure with blocks of size (2n+1)x(2n+1).
    """
    # Total number of coefficients
    L = (N + 1) ** 2
    D = np.zeros((L, L), dtype=np.complex128)

    # Compute for each order n
    for n in range(N + 1):
        # Get the small-d matrix for this order
        d_n = _wigner_small_d(n, beta)

        # Construct the full D matrix for this order
        # D^n_{m',m} = e^{-i m' alpha} * d^n_{m',m}(beta) * e^{-i m gamma}

        m_range = np.arange(-n, n + 1)

        # Phase terms
        # exp(-i * m' * alpha)  [rows]
        phase_left = np.exp(-1j * m_range * alpha)

        # exp(-i * m * gamma)   [cols]
        phase_right = np.exp(-1j * m_range * gamma)

        # Combine: D = diag(phase_left) @ d @ diag(phase_right)
        # Broadcasting: (2n+1, 1) * (2n+1, 2n+1) * (1, 2n+1)
        D_n = phase_left[:, np.newaxis] * d_n * phase_right[np.newaxis, :]

        # Place in the big matrix
        start_idx = n**2
        end_idx = (n + 1) ** 2
        D[start_idx:end_idx, start_idx:end_idx] = D_n

    return D


def _wigner_small_d(j: int, beta: float) -> np.ndarray:
    """
    Compute the Wigner small-d matrix d^j(beta) for a specific order j.

    Uses the explicit sum formula (Edmonds 1957) which is numerically stable
    for the SH orders typically used in audio (N <= 30).

    Parameters
    ----------
    j : int
        SH order (non-negative integer).
    beta : float
        Euler angle in radians.

    Returns
    -------
    d : np.ndarray
        Real Wigner small-d matrix of shape (2j+1, 2j+1).
    """
    dim = 2 * j + 1
    d = np.zeros((dim, dim))

    if j == 0:
        return np.array([[1.0]])

    for m_prime_idx in range(dim):
        m_prime = m_prime_idx - j
        for m_idx in range(dim):
            m = m_idx - j
            val = _calc_d_element(j, m_prime, m, beta)
            d[m_prime_idx, m_idx] = val

    return d


def _calc_d_element(j, mp, m, beta):
    """
    Calculate a single element d^j_{mp, m}(beta) of the Wigner small-d matrix.

    Uses the explicit sum formula (Edmonds 1957):
        d^j_{m',m}(b) = sqrt((j+m')!(j-m')!(j+m)!(j-m)!)
                        * sum_k [(-1)^k / (k! (j-m'-k)! (j+m-k)! (m'-m+k)!)]
                        * (cos b/2)^(2j - m' + m - 2k) * (sin b/2)^(m' - m + 2k)
    """
    c = np.cos(beta / 2)
    s = np.sin(beta / 2)

    factor = np.sqrt(fact(j + mp) * fact(j - mp) * fact(j + m) * fact(j - m))

    res = 0.0

    # Range of k:
    # k >= 0
    # j - mp - k >= 0  => k <= j - mp
    # j + m - k >= 0   => k <= j + m
    # mp - m + k >= 0  => k >= m - mp

    k_min = max(0, m - mp)
    k_max = min(j - mp, j + m)

    for k in range(k_min, k_max + 1):
        denom = fact(k) * fact(j - mp - k) * fact(j + m - k) * fact(mp - m + k)
        term = (
            (-1) ** k * (c ** (2 * j - mp + m - 2 * k)) * (s ** (mp - m + 2 * k))
        ) / denom
        res += term

    return factor * res
