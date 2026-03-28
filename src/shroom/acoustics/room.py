import numpy as np
import pyroomacoustics as pra
from typing import Optional, List, Union, Dict, Tuple
from shroom.acoustics.spatial_signal import SpatialSignal
from shroom.geometry.sampling import sphereicalGrid
from scipy.signal import resample
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
        raise ImportError("Could not import sph_harm or sph_harm_y from scipy.special")

from pyroomacoustics.parameters import constants, Material
from shroom.utils.file_utils import load_file
from shroom.utils.dsp_utils import convolve_multichannel


class Room:
    def __init__(
        self,
        dimensions: Union[List[float], np.ndarray],
        absorption: Optional[Union[float, Dict, pra.Material]] = None,
        materials: Optional[Union[Material, Dict[str, Material]]] = None,
        max_ism_order: int = 10,
        sh_order: int = 14,
        fs: int = 48000,
        sigma2_awgn: Optional[float] = None,
        air_absorption: bool = False,
        temperature: Optional[float] = None,
        humidity: Optional[float] = None,
        ray_tracing: bool = False,
    ):
        """
        Initialize the Room simulation.

        Parameters
        ----------
        dimensions : List[float] or np.ndarray
            Dimensions of the room (x, y, z) in meters.
        absorption : float, dict, or pra.Material, optional
            Absorption characteristics of the walls.
            - If float: Uniform absorption coefficient (0 to 1) for all walls.
            - If dict: Dictionary mapping wall names ('east', 'west', 'north', 'south', 'ceiling', 'floor') to absorption coefficients.
            - If pra.Material: A pyroomacoustics Material object.
            Default is None (must be provided if materials is None).
        materials : pra.Material or dict, optional
            Material properties of the walls.
            - If pra.Material: A single material applied to all walls.
            - If dict: Dictionary mapping wall names to pra.Material objects.
            Mutually exclusive with 'absorption'.
        max_ism_order : int, optional
            Maximum order of reflections for the image source method. Default is 1.
        sh_order : int, optional
            Order of Spherical Harmonics for ARIR computation. Default is 14.
        fs : int, optional
            Sampling frequency. Default is 48000.
        sigma2_awgn : float, optional
            Variance of additive white gaussian noise. Default is None.
        air_absorption : bool, optional
            If True, enables air absorption simulation based on temperature and humidity. Default is False.
        temperature : float, optional
            Temperature in degrees Celsius. Used for speed of sound and air absorption. Default is 20.0.
        humidity : float, optional
            Relative humidity in percent (0-100). Used for air absorption. Default is 50.0.
        ray_tracing : bool, optional
            If True, enables ray tracing for late reverberation (hybrid simulation). Default is False.
        """
        self._validate_inputs(
            dimensions,
            absorption,
            materials,
            max_ism_order,
            fs,
            sigma2_awgn,
            air_absorption,
            temperature,
            humidity,
            ray_tracing,
        )

        self.dimensions = np.asarray(dimensions)
        self.max_order = max_ism_order
        self.sh_order = sh_order
        self.fs = fs
        self.sigma2_awgn = sigma2_awgn
        self.air_absorption = air_absorption
        self.materials = materials
        self.temperature = temperature
        self.humidity = humidity
        self.ray_tracing = ray_tracing

        # Create the pyroomacoustics ShoeBox room
        self.pra_room = pra.ShoeBox(
            self.dimensions,
            fs=self.fs,
            absorption=absorption,
            materials=self.materials,
            max_order=self.max_order,
            sigma2_awgn=self.sigma2_awgn,
            air_absorption=self.air_absorption,
            temperature=self.temperature,
            humidity=self.humidity,
            ray_tracing=self.ray_tracing,
        )

        self.sources = []
        self.receiver_position = None

        self._arirs = None
        self._amb = None
        self.hrtf = None
        self.array = None

        self._remove_dc = True

    @property
    def arirs(self):
        """
        Ambisonics Room Impulse Responses (ARIRs).
        Computed lazily.

        Returns
        -------
        List[SpatialSignal]
            List of ARIRs, one per source.
        """
        if self._arirs is None:
            self._arirs = self.compute_arir()
        return self._arirs

    @property
    def amb(self):
        """
        Ambisonics mix of all sources.
        Computed lazily.

        Returns
        -------
        SpatialSignal
            The mixed Ambisonics signal.
        """
        if self._amb is None:
            self._amb = self.compute_amb()
        return self._amb

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def add_source(
        self,
        position: Union[List[float], np.ndarray],
        signal: Optional[Union[Tuple[np.ndarray, int], np.ndarray, str]] = None,
    ):
        """
        Add a sound source to the room.

        Parameters
        ----------
        position : List[float] or np.ndarray
            Position of the source (x, y, z).
        signal : np.ndarray, tuple, or str, optional
            Source signal (1D array). If None, an impulse is assumed for RIR computation.
        """
        if isinstance(signal, Tuple):
            signal, fs = signal
            if fs != self.fs:
                raise ValueError(f"signal fs ({fs}) does not match Room fs ({self.fs})")
        elif isinstance(signal, str) and signal.endswith(".wav"):
            signal, fs = load_file(signal)
            if fs != self.fs:
                num_samples = int(len(signal) * self.fs / fs)
                signal = resample(signal, num_samples)

        position = np.asarray(position)
        self.pra_room.add_source(position, signal=signal)
        self.sources.append({"position": position, "signal": signal})

    def set_receiver(self, position: Union[List[float], np.ndarray]):
        """
        Set the receiver (listener/microphone array) position.

        Parameters
        ----------
        position : List[float] or np.ndarray
            Position of the receiver (x, y, z).
        """
        self.receiver_position = np.asarray(position)

        # A single dummy microphone is added at the receiver position so that
        # pyroomacoustics computes image sources relative to this location.
        if self.pra_room.mic_array is not None:
            pass

        self.pra_room.add_microphone_array(
            pra.MicrophoneArray(self.receiver_position[:, None], self.fs)
        )

    def compute_arir(self, fdl: int = 81) -> List[SpatialSignal]:
        """
        Compute the Ambisonics Room Impulse Response (ARIR) for all sources.

        Parameters
        ----------
        fdl : int, optional
            Fractional delay length (sinc filter length). Default is 81.

        Returns
        -------
        List[SpatialSignal]
            A list of computed ARIRs (one per source) in the Spherical Harmonics domain (Complex SH).
            Each SpatialSignal has shape: (n_sh_channels, 1, n_samples)
        """
        if self.receiver_position is None:
            raise ValueError("Listener position must be set before computing ARIR.")

        if not self.sources:
            raise ValueError("At least one source must be added.")

        # Run the image source model for all sources
        self.pra_room.image_source_model()

        arirs = []

        for source_idx in range(len(self.sources)):
            arir_signal = self._compute_arir_ism(source_idx, self.sh_order, fdl)
            arirs.append(arir_signal)

        return arirs

    def compute_amb(self) -> SpatialSignal:
        """
        Simulate the room acoustics by convolving source signals with their corresponding ARIRs.

        Returns
        -------
        SpatialSignal
            The final Ambisonics mix of all sources.
            Shape: (n_sh_channels, 1, max_duration_samples)
        """
        if not self.sources:
            raise ValueError("No sources to simulate.")

        mixed_signal = None

        for i, source in enumerate(self.sources):
            signal = source["signal"]
            if signal is None:
                signal = np.array([1.0])

            signal = np.asarray(signal).flatten()

            # Get ARIR for this source
            arir_data = self.arirs[i].data[0, :, :]  # (Channels, Time)
            source_result = convolve_multichannel(signal, arir_data)

            # Accumulate to mix
            if mixed_signal is None:
                mixed_signal = source_result
            else:
                if source_result.shape[1] > mixed_signal.shape[1]:
                    pad_width = (
                        (0, 0),
                        (0, source_result.shape[1] - mixed_signal.shape[1]),
                    )
                    mixed_signal = np.pad(mixed_signal, pad_width)
                elif mixed_signal.shape[1] > source_result.shape[1]:
                    pad_width = (
                        (0, 0),
                        (0, mixed_signal.shape[1] - source_result.shape[1]),
                    )
                    source_result = np.pad(source_result, pad_width)

                mixed_signal += source_result

        # Create output SpatialSignal
        # Reshape to (Channels, Grid(1), Time)
        mixed_signal = mixed_signal[np.newaxis, :, :]

        output_signal = SpatialSignal(
            data=mixed_signal, fs=self.fs, grid=None, is_time=True, is_space=False
        )
        self._amb = output_signal
        return output_signal

    def plot(self, ax=None, plane="xy", extra_obj=None, plot_3d=False):
        """
        Plot the room, sources, and receiver.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        extra_obj : SpatialSignal or similar, optional
            An additional object to visualize orientation for (e.g. Array or HRTF).
            It is assumed to be located at the receiver position.
        plot_3d : bool, optional
            If True, creates a 3D plot. Default is False (2D projection).
        plane : str, optional
            Projection plane for 2D plot: 'xy', 'xz', or 'yz'. Default is 'xy'.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            if plot_3d:
                ax = fig.add_subplot(111, projection="3d")
            else:
                ax = fig.add_subplot(111)

        # Room Dimensions
        L, W, H = self.dimensions

        if plot_3d:
            # 3D Plotting
            # Draw wireframe box
            # Vertices
            r = [0, 1]
            X, Y = np.meshgrid(r, r)
            # Floor
            ax.plot_surface(X * L, Y * W, np.zeros_like(X), alpha=0.1, color="gray")
            # Ceiling
            ax.plot_surface(X * L, Y * W, np.ones_like(X) * H, alpha=0.1, color="gray")

            # Edges
            for x in [0, L]:
                for y in [0, W]:
                    ax.plot([x, x], [y, y], [0, H], "k-", alpha=0.3)
            for z in [0, H]:
                ax.plot([0, L], [0, 0], [z, z], "k-", alpha=0.3)
                ax.plot([0, L], [W, W], [z, z], "k-", alpha=0.3)
                ax.plot([0, 0], [0, W], [z, z], "k-", alpha=0.3)
                ax.plot([L, L], [0, W], [z, z], "k-", alpha=0.3)

            # Sources
            for i, src in enumerate(self.sources):
                pos = src["position"]
                ax.scatter(
                    pos[0],
                    pos[1],
                    pos[2],
                    c="red",
                    marker="x",
                    s=100,
                    label=f"Source {i}" if i == 0 else None,
                )
                ax.text(pos[0], pos[1], pos[2], f" S{i}", verticalalignment="bottom")

            # Receiver
            if self.receiver_position is not None:
                rx = self.receiver_position
                ax.scatter(
                    rx[0], rx[1], rx[2], c="blue", marker="o", s=100, label="Receiver"
                )

                # Orientation
                orientation = None
                if (
                    extra_obj is not None
                    and hasattr(extra_obj, "orientation")
                    and extra_obj.orientation is not None
                ):
                    orientation = extra_obj.orientation
                if orientation is None:
                    orientation = np.array([1.0, 0.0, 0.0])

                orientation = np.asarray(orientation).flatten()
                if orientation.size >= 3:
                    arrow_len = min(L, W, H) * 0.15
                    ax.quiver(
                        rx[0],
                        rx[1],
                        rx[2],
                        orientation[0],
                        orientation[1],
                        orientation[2],
                        length=arrow_len,
                        color="blue",
                        linewidth=2,
                        arrow_length_ratio=0.3,
                    )

            ax.set_xlim(0, L)
            ax.set_ylim(0, W)
            ax.set_zlim(0, H)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")
            ax.set_title("Room Simulation Setup (3D)")

        else:
            # 2D Plotting
            plane = plane.lower()
            if plane == "xy":
                idx_x, idx_y = 0, 1
                dim_x, dim_y = L, W
                labels = ("X [m]", "Y [m]")
                idx_z = 2
            elif plane == "xz":
                idx_x, idx_y = 0, 2
                dim_x, dim_y = L, H
                labels = ("X [m]", "Z [m]")
                idx_z = 1
            elif plane == "yz":
                idx_x, idx_y = 1, 2
                dim_x, dim_y = W, H
                labels = ("Y [m]", "Z [m]")
                idx_z = 0
            else:
                raise ValueError("plane must be 'xy', 'xz', or 'yz'")

            rect = patches.Rectangle(
                (0, 0),
                dim_x,
                dim_y,
                linewidth=2,
                edgecolor="black",
                facecolor="#f0f0f0",
                alpha=0.5,
                label="Room",
            )
            ax.add_patch(rect)

            # Sources
            for i, src in enumerate(self.sources):
                pos = src["position"]
                ax.scatter(
                    pos[idx_x],
                    pos[idx_y],
                    c="red",
                    marker="x",
                    s=100,
                    label=f"Source {i}" if i == 0 else None,
                )
                # Annotate height (the missing dimension)
                ax.text(
                    pos[idx_x],
                    pos[idx_y],
                    f" S{i}\n h={pos[idx_z]:.2f}m",
                    verticalalignment="top",
                    fontsize=9,
                )

            # Receiver
            if self.receiver_position is not None:
                rx = self.receiver_position
                ax.scatter(
                    rx[idx_x], rx[idx_y], c="blue", marker="o", s=100, label="Receiver"
                )
                ax.text(
                    rx[idx_x],
                    rx[idx_y],
                    f" Rx\n h={rx[idx_z]:.2f}m",
                    verticalalignment="top",
                    fontsize=9,
                    color="blue",
                )

                # Orientation
                orientation = None
                if (
                    extra_obj is not None
                    and hasattr(extra_obj, "orientation")
                    and extra_obj.orientation is not None
                ):
                    orientation = extra_obj.orientation
                if orientation is None:
                    orientation = np.array([1.0, 0.0, 0.0])

                orientation = np.asarray(orientation).flatten()
                if orientation.size >= 3:
                    arrow_len = min(dim_x, dim_y) * 0.15
                    # Project orientation to 2D plane
                    u, v = orientation[idx_x], orientation[idx_y]
                    # Normalize projection if significant
                    norm = np.sqrt(u**2 + v**2)
                    if norm > 1e-3:
                        ax.arrow(
                            rx[idx_x],
                            rx[idx_y],
                            u * arrow_len,
                            v * arrow_len,
                            head_width=arrow_len * 0.3,
                            head_length=arrow_len * 0.3,
                            fc="blue",
                            ec="blue",
                            width=arrow_len * 0.05,
                        )

            ax.set_xlim(-0.5, dim_x + 0.5)
            ax.set_ylim(-0.5, dim_y + 0.5)
            ax.set_aspect("equal")
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_title(f"Room Simulation Setup ({plane.upper()} View)")
            ax.grid(True, linestyle="--", alpha=0.3)

        ax.legend(loc="upper right")
        return ax

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _validate_inputs(
        self,
        dimensions: Union[List[float], np.ndarray],
        absorption: Optional[Union[float, Dict, pra.Material]],
        materials: Optional[Union[pra.Material, Dict]],
        max_order: int,
        fs: int,
        sigma2_awgn: Optional[float],
        air_absorption: bool,
        temperature: Optional[float],
        humidity: Optional[float],
        ray_tracing: bool,
    ) -> None:
        dimensions = np.asarray(dimensions)
        if dimensions.shape != (3,):
            raise ValueError(
                f"dimensions must be a 3-element array (x, y, z), got shape {dimensions.shape}"
            )
        if not np.all(dimensions > 0):
            raise ValueError(
                f"Room dimensions must be positive, got {dimensions.tolist()}"
            )

        if materials is not None:
            if not isinstance(materials, (pra.Material, dict)):
                raise TypeError(
                    f"materials must be pra.Material or dict, got {type(materials).__name__}"
                )
            if absorption is not None:
                raise ValueError(
                    "Cannot specify both 'absorption' and 'materials'. Please use only one."
                )

        if absorption is not None:
            if not isinstance(absorption, (float, int, dict, pra.Material)):
                raise TypeError(
                    f"absorption must be float, dict, or pra.Material, got {type(absorption).__name__}"
                )
            if isinstance(absorption, (float, int)) and not (0 <= absorption <= 1):
                raise ValueError(
                    f"Absorption coefficient must be between 0 and 1, got {absorption}"
                )

        if not (isinstance(max_order, (int, np.integer)) and max_order >= 0):
            raise ValueError(
                f"max_ism_order must be a non-negative integer, got {max_order}"
            )
        if not (isinstance(fs, (float, int, np.integer)) and fs > 0):
            raise ValueError(f"fs must be a positive integer or float, got {fs}")

        if sigma2_awgn is not None:
            if not (isinstance(sigma2_awgn, (float, int)) and sigma2_awgn >= 0):
                raise ValueError(
                    f"sigma2_awgn must be a non-negative number, got {sigma2_awgn}"
                )

        if not isinstance(air_absorption, bool):
            raise TypeError(
                f"air_absorption must be a boolean, got {type(air_absorption).__name__}"
            )

        if temperature is not None and not isinstance(temperature, (float, int)):
            raise TypeError(
                f"temperature must be a number, got {type(temperature).__name__}"
            )

        if humidity is not None:
            if not (isinstance(humidity, (float, int)) and 0 <= humidity <= 100):
                raise ValueError(
                    f"humidity must be between 0 and 100, got {humidity}"
                )

        if not isinstance(ray_tracing, bool):
            raise TypeError(
                f"ray_tracing must be a boolean, got {type(ray_tracing).__name__}"
            )

    def _compute_arir_ism(
        self, source_idx: int, sh_order: int, fdl: int
    ) -> "SpatialSignal":
        """
        Internal helper to compute ARIR for a single source using ISM and fractional delays.
        """
        src = self.pra_room.sources[source_idx]

        # 1. Setup Geometry and Constants
        r = self.receiver_position
        c = self.pra_room.c
        fs = self.fs
        fdl2 = fdl // 2

        # Assume all images are visible for now (ShoeBox)
        is_visible = np.ones(src.images.shape[1], dtype=bool)
        images = src.images[:, is_visible]
        att = src.damping[:, is_visible]  # (n_bands, n_images)

        if att.ndim == 1:
            att = att[None, :]

        # 2. Calculate Distances and Delays
        dist = np.sqrt(np.sum((images - r[:, None]) ** 2, axis=0))  # shape (n_images,)
        time = dist / c
        delay = fdl2 / fs
        time += delay  # fractional delay adjustment
        t_max = time.max()

        N = int(np.ceil(t_max * fs + fdl2 + 1)) + 1  # Length of impulse response

        # 3. Apply Attenuations (Distance + Air Absorption)
        oct_band_amplitude = att / (dist + 1e-16)
        if self.pra_room.air_absorption is not None:
            from pyroomacoustics.simulation.ism import apply_air_aborption

            oct_band_amplitude = apply_air_aborption(
                oct_band_amplitude, self.pra_room.air_absorption, dist
            )

        # 4. Compute Fractional Delays
        sample_frac = time * fs
        time_ip = np.floor(sample_frac).astype(np.int32)
        time_fp = (sample_frac - time_ip).astype(np.float32)
        frac_delays = np.zeros((time_fp.shape[0], fdl), dtype=np.float32)

        pra.libroom.fractional_delay(
            frac_delays,
            time_fp,
            constants.get("sinc_lut_granularity"),
            constants.get("num_threads"),
        )

        # 5. Compute Spherical Harmonics and Accumulate
        vecs = images - r[:, None]
        azimuth = np.arctan2(vecs[1], vecs[0])  # phi
        azimuth = np.mod(azimuth, 2 * np.pi)
        colatitude = np.arccos(vecs[2] / (dist + 1e-16))  # theta

        n_sh_channels = (sh_order + 1) ** 2
        arir_data = np.zeros((1, n_sh_channels, N), dtype=np.complex128)

        # --- Vectorised: compute all (n, m) SH values in one broadcast call ---
        # Build index arrays: ns[k] / ms[k] are the degree/order for ACN channel k
        ns_arr = np.array([n for n in range(sh_order + 1)
                           for m in range(-n, n + 1)], dtype=np.int64)[:, None]
        ms_arr = np.array([m for n in range(sh_order + 1)
                           for m in range(-n, n + 1)], dtype=np.int64)[:, None]

        # Y_all: (n_sh_channels, n_images) — all SH values in one call
        Y_all = sph_harm_y(ns_arr, ms_arr,
                           colatitude[None, :], azimuth[None, :]).conj()

        # Effective amplitude per image source: sum bands, then weight by SH
        # oct_band_amplitude: (n_bands, n_images) — sum over bands first
        att_summed = oct_band_amplitude.sum(axis=0)          # (n_images,)
        gains_all  = Y_all * att_summed                      # (n_sh_channels, n_images)

        for i in range(len(time_ip)):
            end = min(time_ip[i] + fdl, N)
            length = end - time_ip[i]
            arir_data[0, :, time_ip[i]:end] += (
                gains_all[:, i:i+1] * frac_delays[i:i+1, :length]
            )

        if self._remove_dc:
            # Enforce zero mean for higher order SH channels (n > 0)
            # This removes non-physical DC components from directional channels
            for n in range(1, sh_order + 1):
                for m in range(-n, n + 1):
                    acn_idx = n**2 + n + m
                    # Subtract mean from time domain signal
                    arir_data[0, acn_idx, :] -= np.mean(arir_data[0, acn_idx, :])

        # 6. Return SpatialSignal
        return SpatialSignal(
            data=arir_data,
            fs=self.fs,
            grid=None,
            is_time=True,
            is_space=False,  # It is in SH domain
        )
