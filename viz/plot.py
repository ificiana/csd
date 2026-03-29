import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.signal import savgol_filter


def plot_3d_trajectory(df, phase_events, window=31, poly=3):
    df = df.sort_index()
    t = df.index.values
    x, y, z = df["x"].values, df["y"].values, df["z"].values

    # Smooth
    x_s = savgol_filter(x, window, poly)
    y_s = savgol_filter(y, window, poly)
    z_s = savgol_filter(z, window, poly)

    # Style
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    norm = plt.Normalize(t.min(), t.max())
    cmap = cm.viridis

    for i in range(len(t) - 1):
        ax.plot(
            x_s[i : i + 2],
            y_s[i : i + 2],
            z_s[i : i + 2],
            color=cmap(norm(t[i])),
            linewidth=2,
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(r"$t$ [s]", fontdict={"fontsize": 14})

    # Labels
    ax.set_xlabel(r"$x$ [m]", fontdict={"fontsize": 14})
    ax.set_ylabel(r"$y$ [m]", fontdict={"fontsize": 14})
    ax.set_zlabel(r"$z$ [m]", fontdict={"fontsize": 14})

    # Phase annotations
    for t_event, label in phase_events:
        idx = np.argmin(np.abs(t - t_event))
        ax.text(x_s[idx], y_s[idx], z_s[idx], label, fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_position_vs_time(df, phase_events, window=31, poly=3):

    # ------------------------------------------------------------
    # 1. DATA PREP
    # ------------------------------------------------------------
    df = df.sort_index()
    t = df.index.values

    dims = ["x", "y", "z"]

    smooth, std = {}, {}

    for d in dims:
        v = df[d].values
        s = savgol_filter(v, window, poly)

        resid = v - s
        sigma = (
            pd.Series(resid).rolling(window, center=True).std().bfill().ffill().values
        )

        smooth[d] = s
        std[d] = sigma

    # ------------------------------------------------------------
    # 2. STYLE (stronger grid, cleaner typography)
    # ------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ylabels = [r"$x(t)$ [m]", r"$y(t)$ [m]", r"$z(t)$ [m]"]
    # panels = ["(b)", "(c)", "(d)"]

    # ------------------------------------------------------------
    # 3. PLOTTING
    # ------------------------------------------------------------
    for ax, d, yl in zip(axes, dims, ylabels):
        s = smooth[d]
        sig = std[d]

        ax.plot(t, s, lw=2)
        ax.fill_between(t, s - sig, s + sig, alpha=0.2)

        ax.set_ylabel(yl, fontdict={"fontsize": 14})

    # ------------------------------------------------------------
    # 4. PHASE ANNOTATION (clean + global)
    # ------------------------------------------------------------
    for i in range(len(phase_events) - 1):
        t0, label = phase_events[i]
        t1, _ = phase_events[i + 1]

        for ax in axes:
            ax.axvspan(t0, t1, color="gray", alpha=0.06)

        # place label only on top axis (avoids clutter)
        axes[0].text(
            (t0 + t1) / 2,
            1.02,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # phase boundaries (stronger visual cue)
    for t_event, _ in phase_events:
        for ax in axes:
            ax.axvline(t_event, linestyle="--", alpha=0.5, linewidth=1)

    # ------------------------------------------------------------
    # 5. FINALIZE
    # ------------------------------------------------------------
    axes[-1].set_xlabel(r"$t$ [s]", fontdict={"fontsize": 14})

    plt.tight_layout()
    plt.show()


def plot_attitude_vs_time(df, phase_events, window=31, poly=3):
    # ------------------------------------------------------------
    # 1. DATA PREP
    # ------------------------------------------------------------
    df = df.sort_index()
    t = df.index.values

    dims = ["roll", "pitch", "yaw"]

    smooth, std = {}, {}

    for d in dims:
        v = df[d].values * np.pi / 180  # convert to radians
        s = savgol_filter(v, window, poly)

        resid = v - s
        sigma = (
            pd.Series(resid).rolling(window, center=True).std().bfill().ffill().values
        )

        smooth[d] = s
        std[d] = sigma

    # ------------------------------------------------------------
    # 2. STYLE (stronger grid, cleaner typography)
    # ------------------------------------------------------------
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.grid": True,
            "grid.alpha": 0.4,
            "grid.linewidth": 0.8,
            "grid.linestyle": "--",
        }
    )

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    ylabels = [
        r"Roll: $\phi(t)$ [rad]",
        r"Pitch: $\theta(t)$ [rad]",
        r"Yaw: $\psi(t)$ [rad]",
    ]
    # panels = ["(b)", "(c)", "(d)"]

    # ------------------------------------------------------------
    # 3. PLOTTING
    # ------------------------------------------------------------
    for ax, d, yl in zip(axes, dims, ylabels):
        s = smooth[d]
        sig = std[d]

        ax.plot(t, s, lw=2)
        ax.fill_between(t, s - sig, s + sig, alpha=0.2)

        ax.set_ylabel(yl, fontdict={"fontsize": 14})

    # ------------------------------------------------------------
    # 4. PHASE ANNOTATION (clean + global)
    # ------------------------------------------------------------
    for i in range(len(phase_events) - 1):
        t0, label = phase_events[i]
        t1, _ = phase_events[i + 1]

        for ax in axes:
            ax.axvspan(t0, t1, color="gray", alpha=0.06)

        # place label only on top axis (avoids clutter)
        axes[0].text(
            (t0 + t1) / 2,
            1.02,
            label,
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # phase boundaries (stronger visual cue)
    for t_event, _ in phase_events:
        for ax in axes:
            ax.axvline(t_event, linestyle="--", alpha=0.5, linewidth=1)

    # ------------------------------------------------------------
    # 5. FINALIZE
    # ------------------------------------------------------------
    axes[-1].set_xlabel(r"$t$ [s]", fontdict={"fontsize": 14})

    plt.tight_layout()
    plt.show()
