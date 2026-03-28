import numpy as np
from typing import Optional


def magls(
    A: np.ndarray,
    b: np.ndarray,
    x_prev: np.ndarray,
    A_prev: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    L: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
    iters: int = 1,
) -> np.ndarray:
    """
    Approximate the solution to the Magnitude Least Squares problem:
    min_x || |A x| - |b| ||_2^2 + lam * ||L x||_2^2

    This is solved iteratively by linearizing the phase term:
    b_new = |b| * exp(j * angle(A_prev * x_prev))
    min_x || A x - b_new ||_2^2 + lam * ||L x||_2^2

    Parameters
    ----------
    A : np.ndarray
        System matrix.
    b : np.ndarray
        Target magnitude vector (complex or real).
    x_prev : np.ndarray
        Previous estimate of x.
    A_prev : np.ndarray, optional
        Previous system matrix (if different). Default is A.
    lam : float, optional
        Regularization parameter.
    L : np.ndarray, optional
        Regularization matrix.
    rcond : float, optional
        Cutoff for singular values.
    iters : int, optional
        Number of iterations. Default is 1.

    Returns
    -------
    x : np.ndarray
        Estimated solution.
    """
    for iter in range(iters):
        if A_prev is None:
            A_prev = A
        phi = np.angle(np.dot(A_prev, x_prev))
        b_new = np.abs(b) * np.exp(1j * phi)
        x = tikhonov(A, b_new, lam=lam, L=L, rcond=rcond)
        x_prev = x
    return x


def regularized_pinv(
    A: np.ndarray,
    lam: Optional[float] = None,
    L: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> np.ndarray:
    """
    Computes the regularized pseudo-inverse of a matrix A using Tikhonov regularization.

    Parameters
    ----------
    A : np.ndarray
        The input matrix (M x N).
    lam : float, optional
        Regularization parameter (lambda). If None, it is estimated from the largest singular value.
    L : np.ndarray, optional
        Tikhonov regularization matrix (default is identity).
    rcond : float, optional
        Cutoff for small singular values (not used in this implementation but kept for signature compatibility).

    Returns
    -------
    pinv_reg : np.ndarray
        The regularized pseudo-inverse of A.
    """
    A = np.asarray(A)
    m, n = A.shape

    if lam is None:
        lam = _estimate_lambda(A)

    if L is None:
        L = np.eye(n, dtype=A.dtype)
    else:
        L = np.asarray(L)
        if L.shape[1] != n:
            raise ValueError(f"L must have shape (P, {n}), got {L.shape}")

    # Term: (A^H A + lambda^2 * L^H L)
    regularization_term = (lam ** 2) * np.dot(L.conj().T, L)
    gram_matrix = np.dot(A.conj().T, A) + regularization_term

    # Solve for the inverse operator: (Gram)^-1 @ A^H
    pinv_reg = np.linalg.solve(gram_matrix, A.conj().T)

    return pinv_reg

def _estimate_lambda(A: np.ndarray) -> np.floating:
    """Estimates the regularization parameter lambda based on the largest singular value."""
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = s[0] if s.size > 0 else 1.0
    return 0.01 * sigma_max


def tikhonov(
    A: np.ndarray,
    b: np.ndarray,
    lam: Optional[float] = None,
    L: Optional[np.ndarray] = None,
    rcond: Optional[float] = None,
) -> np.ndarray:
    """
    Solve (A^H A + lam^2 L^H L) x = A^H b via Tikhonov regularization.

    If lam is None, it is automatically selected as a fraction of the
    largest singular value of A (heuristic).

    Parameters
    ----------
    A : array_like, shape (M, N)
        System matrix.
    b : array_like, shape (M,) or (M, K)
        Right-hand side.
    lam : float or None
        Regularization parameter (lambda). If None, calculated automatically.
    L : array_like or None, shape (P, N), optional
        Regularization matrix. If None, L = I (standard ridge).
    rcond : float or None, optional
        Cutoff for small singular values, passed to np.linalg.lstsq.

    Returns
    -------
    x : ndarray, shape (N,) or (N, K)
        Regularized solution.
    """
    A = np.asarray(A)
    b = np.asarray(b)
    m, n = A.shape

    # 1. Automatic Lambda Selection (Heuristic)
    # ---------------------------------------------------------
    if lam is None:
        # Calculate largest singular value of A to determine scale
        # We use compute_uv=False because we only need the values (faster)
        s = np.linalg.svd(A, compute_uv=False)
        sigma_max = s[0] if s.size > 0 else 1.0
        sigma_min = s[-1] if s.size > 1 else sigma_max
        # Floor sigma_min to prevent inf condition number when A is rank-deficient
        # (e.g. at DC where the steering matrix collapses to rank-1).
        sigma_min = max(sigma_min, 1e-6 * sigma_max)
        condition_number = sigma_max / sigma_min

        lam = max(1e-7 * condition_number, 1e-12)
        # lam = min(lam, 1e-2 * sigma_max)  # cap: never over-regularize beyond 1% of signal scale


    # 2. Setup Regularization Matrix L
    # ---------------------------------------------------------
    if L is None:
        L = np.eye(n, dtype=A.dtype)
    else:
        L = np.asarray(L)
        if L.shape[1] != n:
            raise ValueError(f"L must have shape (P, {n}), got {L.shape}")

    # 3. Build Augmented System
    #    Minimize ||Ax - b||^2 + ||lam * L x||^2
    #    Equivalent to solving:
    #    [A      ] x = [b]
    #    [lam * L]     [0]
    # ---------------------------------------------------------

    # Note: Changed from np.sqrt(lam) to lam to match docstring formula (lam^2 L^H L)
    lamL = lam * L

    if b.ndim == 1:
        A_aug = np.vstack([A, lamL])
        b_aug = np.concatenate([b, np.zeros(L.shape[0], dtype=b.dtype)])
    else:
        A_aug = np.vstack([A, lamL])
        zeros_block = np.zeros((L.shape[0], b.shape[1]), dtype=b.dtype)
        b_aug = np.vstack([b, zeros_block])

    # 4. Solve Least Squares
    x, *_ = np.linalg.lstsq(A_aug, b_aug, rcond=rcond)
    return x