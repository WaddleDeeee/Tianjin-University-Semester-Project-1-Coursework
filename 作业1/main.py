"""Plot a 4-second tongue motor-imagery trial from subject A04T.

Dependencies: numpy, scipy, matplotlib
Run in PyCharm, or: python main.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


# BCI Competition IV 2a / BNCI 2014-001 EEG channel order.
CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
SUBJECT = 4
CLASS_ID = 4                 # 1: left hand, 2: right hand, 3: feet, 4: tongue
CUE_DELAY_SECONDS = 2.0     # The cue appears 2 seconds after trial start.
DURATION_SECONDS = 4.0


def load_first_clean_tongue_trial(data_file):
    """Return samples x EEG channels, sampling rate, run number, and trial number."""
    contents = loadmat(
        data_file, struct_as_record=False, squeeze_me=True, variable_names=["data"]
    )
    for run_number, run in enumerate(np.atleast_1d(contents["data"]), start=1):
        labels = np.asarray(run.y).reshape(-1)
        starts = np.asarray(run.trial).reshape(-1)
        artifacts = np.asarray(run.artifacts).reshape(-1)
        fs = int(run.fs)

        # Calibration runs have no class labels and are skipped automatically.
        for trial_index in np.flatnonzero(labels == CLASS_ID):
            if artifacts.size and artifacts[trial_index] != 0:
                continue

            # MATLAB uses 1-based sample positions; Python uses 0-based indices.
            trial_start = int(starts[trial_index]) - 1
            start = trial_start + round(CUE_DELAY_SECONDS * fs)
            stop = start + round(DURATION_SECONDS * fs)
            if start < 0 or stop > run.X.shape[0]:
                continue

            # The first 22 columns are EEG; the last 3 EOG columns are excluded.
            eeg = np.asarray(run.X[start:stop, :22], dtype=float)
            if eeg.shape[1] != 22 or not np.isfinite(eeg).all():
                continue
            return eeg, fs, run_number, int(trial_index) + 1

    raise ValueError("No complete, unflagged tongue trial was found in this file.")


def plot_eeg(eeg, fs, run_number, trial_number):
    # Remove each channel's mean for display, preserving its amplitude scale.
    centered = eeg - eeg.mean(axis=0, keepdims=True)
    gap = max(60.0, float(np.ptp(centered, axis=0).max()) * 1.2)
    offsets = np.arange(21, -1, -1) * gap
    time = np.arange(eeg.shape[0]) / fs

    fig, ax = plt.subplots(figsize=(13, 12))
    for channel, offset in enumerate(offsets):
        ax.axhline(offset, color="#e3e9ef", linewidth=0.6, zorder=0)
        ax.plot(time, centered[:, channel] + offset, color="#287eae", linewidth=0.7)

    ax.set_yticks(offsets, CHANNEL_NAMES)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", length=0, pad=12, labelsize=9)
    ax.set_xticks(np.arange(0, DURATION_SECONDS + 0.1, 0.5))
    ax.set_xlim(-0.07, DURATION_SECONDS + 0.2)
    ax.set_ylim(-gap, offsets[0] + gap)
    ax.set_xlabel("Time since tongue-imagery cue (s)", fontsize=11)
    ax.grid(axis="x", color="#dce4eb", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="#28865b", linestyle="--", linewidth=1)
    ax.axvline(DURATION_SECONDS, color="#b85f55", linestyle="--", linewidth=1)

    # Physical amplitude scale: signals in this MAT release are in microvolts.
    scale_x, scale_y, scale_uv = 4.1, -0.65 * gap, 20.0
    ax.plot([scale_x, scale_x], [scale_y, scale_y + scale_uv], color="#34495e", lw=1.5)
    ax.plot([scale_x - 0.025, scale_x + 0.025], [scale_y, scale_y], color="#34495e", lw=1.5)
    ax.plot([scale_x - 0.025, scale_x + 0.025], [scale_y + scale_uv] * 2, color="#34495e", lw=1.5)
    ax.text(scale_x - 0.06, scale_y + scale_uv / 2, "20 µV", ha="right", va="center", fontsize=9)

    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    fig.suptitle("Subject 04 | Tongue motor imagery | 22 EEG channels", fontsize=17, y=0.975)
    ax.set_title(
        f"A04T.mat | run {run_number}, trial {trial_number} | {fs} Hz | "
        f"{eeg.shape[0]} samples/channel | trial time 2–6 s",
        fontsize=10, color="#596775", pad=14,
    )
    fig.text(
        0.5, 0.025,
        "Per-channel mean removed for display; no additional filtering. EOG channels excluded.",
        ha="center", fontsize=9, color="#596775",
    )
    fig.subplots_adjust(left=0.055, right=0.92, bottom=0.085, top=0.92)
    return fig


def main():
    project_dir = Path(__file__).resolve().parent
    default_data = project_dir / "archive" / "A04T.mat"
    if not default_data.is_file():
        default_data = Path.home() / "Desktop" / "BCI_Signal_processing" / "archive" / "A04T.mat"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=default_data)
    parser.add_argument("--output", type=Path, default=project_dir / "outputs" / "subject04_tongue_4s.png")
    parser.add_argument("--no-show", action="store_true", help="Save without opening a plot window")
    args = parser.parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"MAT file not found: {args.data}. Use --data to set its location.")

    eeg, fs, run_number, trial_number = load_first_clean_tongue_trial(args.data)
    print(f"File: {args.data}")
    print(f"Run {run_number}, trial {trial_number}: tongue, no source artifact flag")
    print(f"EEG shape: {eeg.shape}; sampling rate: {fs} Hz")
    print("Window: trial start + 2 s to + 6 s (0–4 s after cue)")
    fig = plot_eeg(eeg, fs, run_number, trial_number)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    print(f"Saved plot: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
