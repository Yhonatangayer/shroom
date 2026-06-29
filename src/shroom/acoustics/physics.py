import numpy as np
from scipy.special import spherical_jn, spherical_yn

# --- Physical Constants ---
SPEED_OF_SOUND = 343.0  # m/s
DEFAULT_SOURCE_DISTANCE = 2.0  # m (for array manifold calculation)


def _compute_bn_diagonal(
    N: int,
    k: np.ndarray,
    a: float,
    r_m: float,
    sphere_type: str = "rigid",
    source_type: str = "plane_wave",
    r_s: float = 2.0,
    apply_damping: bool = False,
) -> np.ndarray:
    """
    Computes the diagonal matrix of radial functions (Bn) for a spherical array.

    Parameters
    ----------
    N : int
        Maximum order of spherical harmonics.
    k : np.ndarray
        Wave numbers (frequency bins).
    a : float
        Radius of the sphere.
    r_m : float
        Radius of the microphone.
    sphere_type : str, optional
        Type of the sphere ('rigid' or 'open'). Default is 'rigid'.
    source_type : str, optional
        Source model: 'plane_wave' (default) or 'point_source'.
        'plane_wave' uses Bn = 4π·iⁿ·b_n(k), which matches the standard
        spherical array literature and is independent of source distance.
        'point_source' uses the full Green's function expansion with a
        point source at distance r_s.
    r_s : float, optional
        Distance of the point source in metres. Only used when
        source_type='point_source'. Default is 2.0.
    apply_damping : bool, optional
        If True, applies smooth Wiener-style magnitude damping to the radial
        functions to improve numerical stability (see ``_apply_magnitude_damping``).
        Default is False. Note: a previous per-order sigmoid "order mask" was
        removed because it rang in the time domain (audible ghost image); the
        magnitude knee alone provides the same stabilisation smoothly.

    Returns
    -------
    bn_diag : np.ndarray
        The diagonal matrix of radial functions.
    """
    k = np.asarray(k)
    F = k.size
    L = (N + 1) ** 2

    # Preallocate bn per order: bn_n[n, f]
    bn_n = np.zeros((N + 1, F), dtype=np.complex128)

    # Avoid k=0 numerical issues
    k_safe = np.where(k == 0.0, 1e-10, k)

    for n in range(N + 1):
        bn_n[n, :] = _compute_bn_for_order(n, k_safe, a, r_m, sphere_type, source_type, r_s)

    # Apply damping and masking
    if apply_damping:
        bn_n = _apply_magnitude_damping(bn_n)

    # Handle DC (k=0): force n=0 real, zero n>=1 (j_n(0)=0 for n>=1 exactly at DC)
    dc_mask = k == 0.0
    bn_n[0, dc_mask] = bn_n[0, dc_mask].real + 0.0j
    bn_n[1:, dc_mask] = 0.0

    # Expand per (n, m)
    bn_diag = _expand_bn_to_diagonal(bn_n, N, L, F)

    return bn_diag


def _compute_bn_for_order(
    n: int,
    k: np.ndarray,
    a: float,
    r_m: float,
    sphere_type: str,
    source_type: str = "plane_wave",
    r_s: float = 2.0,
) -> np.ndarray:
    """Computes the radial function for a specific order n."""
    jn_krm = spherical_jn(n, k * r_m)
    yn_krm = spherical_yn(n, k * r_m)
    hn2_krm = jn_krm - 1j * yn_krm

    if sphere_type == "open" or a == 0.0:
        scattering_term = jn_krm
    else:
        jn_der_ka = spherical_jn(n, k * a, derivative=True)
        yn_der_ka = spherical_yn(n, k * a, derivative=True)
        hn2_der_ka = jn_der_ka - 1j * yn_der_ka
        ratio = jn_der_ka / hn2_der_ka
        scattering_term = jn_krm - hn2_krm * ratio

    if source_type == "plane_wave":
        return 4 * np.pi * (1j ** n) * scattering_term
    else:
        jn_krs = spherical_jn(n, k * r_s)
        yn_krs = spherical_yn(n, k * r_s)
        hn2_krs = jn_krs - 1j * yn_krs
        return 4 * np.pi * (-1j) * (k * r_s) * hn2_krs * scattering_term


def _apply_magnitude_damping(bn_n: np.ndarray) -> np.ndarray:
    """Applies smooth magnitude damping to the radial functions.

    Uses a Wiener-style gain ``|b_n|^2 / (|b_n|^2 + limit^2)`` that smoothly
    suppresses numerically-insignificant radial coefficients (which dominate at
    low frequencies for high orders) without amplifying them in the inverse.

    A previous version also applied a per-order sigmoid order mask
    ``1 / (1 + exp(n - (ka + 1)))``. That mask is a staircase of ~N sharp
    spectral edges in frequency, which rings in the time domain and produces an
    audible "ghost"/duplicate image when the radial filters are long (large
    ``duration`` => fine frequency resolution => the ringing extends past the
    ~10 ms echo-fusion window). The magnitude knee below already removes the same
    insignificant coefficients smoothly, so the order mask is omitted.
    """
    limit = 1e-4
    mag_sq = np.abs(bn_n) ** 2
    damping = mag_sq / (mag_sq + limit**2)
    bn_n *= damping

    return bn_n


def _expand_bn_to_diagonal(bn_n: np.ndarray, N: int, L: int, F: int) -> np.ndarray:
    """Expands the radial functions to the full diagonal matrix."""
    bn_diag = np.zeros((L, F), dtype=np.complex128)
    for n in range(N + 1):
        start = n * n
        end = (n + 1) * (n + 1)
        bn_diag[start:end, :] = bn_n[n, :][np.newaxis, :]
    return bn_diag
