import os
from itertools import cycle

import numpy as np
from matplotlib import pyplot as plt


def loglog_plot(
    freqs,
    errors: dict,
    figsize=(10, 5),
    title=None,
    save_path=None,
    show=True,
    styles: dict = None,
    colors: dict = None,
    ylabel="Error (dB)",
    xlim=None,
    ylim=None,
    beta=0.1,
    variances: dict = None,
):
    """
    Plot error curves. Automatically changes line style if a curve overlaps
    significantly with a previously plotted one.

    Parameters
    ----------
    styles : dict, optional
        Map label → linestyle string (e.g. ``{"curve": "--"}``).  When given
        for a label the auto-overlap detection is skipped for that curve.
    colors : dict, optional
        Map label → matplotlib color (e.g. ``{"curve": "red"}`` or
        ``{"curve": "#1f77b4"}``).  Can be combined freely with ``styles``;
        labels not present keep the default matplotlib color cycle.
    beta : float, optional (default=0.1)
        Overlap threshold in dB. If the maximum difference between a new curve
        and any previous curve is less than 'beta', the new curve is considered
        "on top" and its style is changed.
    variances : dict, optional
        Same keys as `errors`. Each value is a 1-D array of variance values
        (same units as the corresponding error array). When provided for a key,
        a ±1 std shaded band is drawn around the mean curve in dB.
    """
    plt.figure(figsize=figsize)

    # Store the dB values of curves we have already plotted to check for overlaps
    history_db = []

    # Cycle of styles to use ONLY when overlap is detected
    # (Dashed, Dotted, Dash-Dot)
    overlap_style_cycler = cycle(["--", ":", "-."])

    for label, err in errors.items():
        # 1. Convert to dB for plotting and comparison
        curr_db = 10 * np.log10(err)

        # 2. Determine linestyle
        # Priority 1: User manual styles
        if styles is not None and label in styles:
            line_style = styles[label]

        else:
            # Priority 2: Check for overlap with ANY previous curve
            is_overlapping = False
            for prev_db in history_db:
                # Check if the curves are "on top of each other" (max diff < beta)
                # You can change np.max to np.mean if you want a looser 'average' check
                if np.max(np.abs(curr_db - prev_db)) < beta:
                    is_overlapping = True
                    break

            if is_overlapping:
                line_style = next(overlap_style_cycler)
            else:
                line_style = "-"  # Default solid for unique curves

        # 3. Determine color (None → matplotlib default cycle)
        color = colors[label] if (colors is not None and label in colors) else None

        # 4. Plot — capture the line to reuse its color for the variance band
        plot_kwargs = dict(label=label, linestyle=line_style)
        if color is not None:
            plot_kwargs["color"] = color
        (line,) = plt.plot(freqs, curr_db, **plot_kwargs)

        # 5. Variance shading: ±1 std band around the mean curve in dB.
        # The spread is computed from the upper half (err + std) and mirrored
        # symmetrically, so the band never collapses to -inf when err - std <= 0.
        if variances is not None and label in variances:
            std = np.sqrt(np.maximum(variances[label], 0.0))
            upper_db = 10 * np.log10(np.maximum(err + std, 1e-20))
            db_spread = upper_db - curr_db          # always >= 0
            lower_db = curr_db - db_spread
            plt.fill_between(
                freqs, lower_db, upper_db,
                alpha=0.2, color=line.get_color(), linewidth=0,
            )

        # 6. Save to history
        history_db.append(curr_db)

    if ylim is not None:
        plt.ylim(ylim)

    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel(ylabel)
    if title is not None:
        plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale("log")
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Plot saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()
