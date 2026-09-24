"""ARIR generation: SHroom vs. pyroomacoustics spherical-harmonic receivers.

Both pipelines share the same image-source engine (``pra.ShoeBox``). The pyroomacoustics
baseline places one microphone per SH channel at the receiver, each with a
``RealSphericalHarmonicsDirectivity`` (ACN order), and calls ``compute_rir``. SHroom
projects all image sources onto the SH basis in a single batched step
(``Room.compute_arir``).

For each SH order the script reports
  * the wall-clock time of both pipelines (ISM included, median of ``N_REPEATS``), and
  * the relative error between the two ARIRs, after mapping SHroom's complex SH
    output to pyroomacoustics' real-SH convention with the exact basis transform.

For a like-for-like comparison, SHroom's randomized ISM and pyroomacoustics' 10 Hz RIR
high-pass filter are both disabled; the two ARIRs then agree to float32 precision.

Requires pyroomacoustics >= 0.10 (``RealSphericalHarmonicsDirectivity``).
"""

import time
import warnings

import numpy as np
import pyroomacoustics as pra
from pyroomacoustics.directivities.harmonics import (
    RealSphericalHarmonicsDirectivity,
    get_mn_in_acn_order,
    real_sph_harm,
)

from shroom.acoustics.room import Room, sph_harm_y

FS = 16000
ROOM_DIMS = [6.0, 5.0, 3.0]
ABSORPTION = 0.4  # energy absorption coefficient
MAX_ISM_ORDER = 10
SOURCE_POS = [4.0, 3.5, 1.7]
RECEIVER_POS = [2.0, 2.0, 1.5]
SH_ORDERS = [1, 2, 4, 6, 8, 10, 12]
N_REPEATS = 3
FDL = 81  # fractional-delay filter length, the default of both libraries


def complex_to_real_sh(order: int) -> np.ndarray:
    """Matrix T with Y_real = T @ conj(Y_complex), both in ACN order.

    Y_complex is SciPy's complex SH (as conjugated by SHroom's ARIR) and Y_real is
    pyroomacoustics' real SH. T is fitted on random directions; it is exact because
    both are bases of the same space of order-``order`` spherical functions.
    """
    rng = np.random.default_rng(0)
    n_dirs = 4 * (order + 1) ** 2
    az = rng.uniform(0, 2 * np.pi, n_dirs)
    co = np.arccos(rng.uniform(-1, 1, n_dirs))
    ms, ns = get_mn_in_acn_order(order)
    y_c = sph_harm_y(ns[:, None], ms[:, None], co[None, :], az[None, :]).conj()
    y_r = np.stack([real_sph_harm(n, m, co, az) for m, n in zip(ms, ns)])
    t, *_ = np.linalg.lstsq(y_c.T, y_r.T, rcond=None)
    return t.T


def shroom_arir(order: int) -> np.ndarray:
    room = Room(
        dimensions=ROOM_DIMS,
        absorption=ABSORPTION,
        max_ism_order=MAX_ISM_ORDER,
        sh_order=order,
        fs=FS,
        use_rand_ism=False,  # plain ISM, as in the pyroomacoustics baseline
    )
    room._remove_dc = False  # pyroomacoustics applies no DC removal
    room.add_source(SOURCE_POS)
    room.set_receiver(RECEIVER_POS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ISM-coverage warning is irrelevant here
        return room.compute_arir()[0].data[0]  # (n_sh, T), complex SH


def pra_arir(order: int) -> np.ndarray:
    pra.constants.set("rir_hpf_enable", False)  # SHroom applies no high-pass filter
    room = pra.ShoeBox(
        ROOM_DIMS, fs=FS, materials=pra.Material(ABSORPTION), max_order=MAX_ISM_ORDER
    )
    room.add_source(SOURCE_POS)
    for m, n in zip(*get_mn_in_acn_order(order)):
        room.add_microphone(
            RECEIVER_POS, directivity=RealSphericalHarmonicsDirectivity(m, n)
        )
    room.compute_rir()
    return np.stack([rir[0] for rir in room.rir])  # (n_sh, T), real SH


def timed(fn, order: int):
    times, out = [], None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        out = fn(order)
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), out


def arir_error(h_shroom: np.ndarray, h_pra: np.ndarray, order: int) -> float:
    """Relative L2 error ||h_shroom - h_pra|| / ||h_pra|| after basis mapping.

    SHroom's ARIR carries an extra bulk delay of ``FDL // 2`` samples (half the
    fractional-delay filter), removed before the sample-by-sample comparison.
    No gain is fitted: both libraries use the same 1/r amplitude.
    """
    h_s = (complex_to_real_sh(order) @ h_shroom).real[:, FDL // 2 :]
    n = min(h_s.shape[1], h_pra.shape[1])
    h_s, h_p = h_s[:, :n], h_pra[:, :n]
    return float(np.linalg.norm(h_s - h_p) / np.linalg.norm(h_p))


def main():
    print(
        f"Room {ROOM_DIMS} m, energy absorption {ABSORPTION}, ISM order "
        f"{MAX_ISM_ORDER}, fs {FS} Hz, median of {N_REPEATS} runs\n"
    )
    print(f"{'N':>3} {'channels':>9} {'pra [s]':>9} {'shroom [s]':>11} "
          f"{'speed-up':>9} {'rel. error':>11}")
    for order in SH_ORDERS:
        t_pra, h_pra = timed(pra_arir, order)
        t_shroom, h_shroom = timed(shroom_arir, order)
        err = arir_error(h_shroom, h_pra, order)
        print(f"{order:>3} {(order + 1) ** 2:>9} {t_pra:>9.3f} {t_shroom:>11.3f} "
              f"{t_pra / t_shroom:>8.1f}x {err:>11.2e}")


if __name__ == "__main__":
    main()
