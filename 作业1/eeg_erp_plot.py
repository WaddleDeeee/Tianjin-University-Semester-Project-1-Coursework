"""Subject 04: continuous EEG, task cues, and a tongue cue-locked average.

Dependencies: numpy, scipy, matplotlib
Run: python eeg_erp_plot.py
The average is across repeated trials of the same channel, not across channels.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
CLASS_NAMES = {1: "Left hand", 2: "Right hand", 3: "Feet", 4: "Tongue"}


def prepare_data(path, channel):
    runs = np.atleast_1d(loadmat(
        path, struct_as_record=False, squeeze_me=True, variable_names=["data"]
    )["data"])
    channel_index = CHANNEL_NAMES.index(channel)
    epochs = []
    first_run = None
    sample_rate = None
    flagged = 0
    incomplete = 0
    for run_number, run in enumerate(runs, 1):
        labels = np.asarray(run.y).reshape(-1).astype(int)
        if labels.size == 0:
            continue
        fs = int(run.fs)
        if sample_rate is not None and fs != sample_rate:
            raise ValueError("Sampling rates differ between runs.")
        sample_rate = fs
        # Trial positions are 1-based in MATLAB; the task cue is 2 seconds later.
        cues = np.asarray(run.trial).reshape(-1).astype(int) - 1 + 2 * fs
        flags = np.asarray(run.artifacts).reshape(-1)
        if first_run is None:
            first_run = (run_number, run, cues, labels)
        before, after = round(0.2 * fs), round(0.8 * fs)
        for i in np.flatnonzero(labels == 4):
            if flags.size and flags[i] != 0:
                flagged += 1
                continue
            start, stop = cues[i] - before, cues[i] + after + 1
            if start < 0 or stop > run.X.shape[0]:
                incomplete += 1
                continue
            epoch = np.array(run.X[start:stop, channel_index], dtype=float, copy=True)
            if not np.isfinite(epoch).all():
                incomplete += 1
                continue
            # Subtract the pre-cue mean (-200 ms <= t < 0 ms) from this trial.
            epoch -= epoch[:before].mean()
            epochs.append(epoch)
    if not epochs:
        raise ValueError("No valid tongue epochs found.")
    times_ms = np.arange(-before, after + 1) / sample_rate * 1000
    return np.stack(epochs), times_ms, first_run, sample_rate, flagged, incomplete


def draw_figure(epochs, times_ms, first_run, fs, channel):
    run_number, run, cues, labels = first_run
    count = min(round(35 * fs), run.X.shape[0])
    continuous = np.array(run.X[:count, CHANNEL_NAMES.index(channel)], copy=True)
    continuous -= continuous.mean()  # Display offset only; no added filtering.
    time_s = np.arange(count) / fs
    fig = plt.figure(figsize=(13, 9.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.25, 0.4, 2.1], hspace=0.38)
    raw_ax = fig.add_subplot(grid[0])
    cue_ax = fig.add_subplot(grid[1], sharex=raw_ax)
    erp_ax = fig.add_subplot(grid[2])

    raw_ax.plot(time_s, continuous, color="#446078", lw=0.65)
    raw_ax.set_title(f"A. Continuous EEG | first 35 s of run {run_number}", loc="left", fontsize=12)
    raw_ax.set_ylabel("Amplitude (µV)")
    raw_ax.tick_params(labelbottom=False)
    raw_ax.set_xlim(0, count / fs)
    raw_ax.grid(alpha=0.15)
    cue_ax.set_title("B. Actual task-cue times", loc="left", fontsize=11, pad=3)
    cue_ax.set_ylim(0, 1.2)
    cue_ax.set_yticks([])
    cue_ax.set_xlabel("Time in this run (s)")
    for cue, label in zip(cues, labels):
        seconds = cue / fs
        if not 0 <= seconds < count / fs:
            continue
        color = "#13866b" if label == 4 else "#8795a3"
        raw_ax.axvline(seconds, color=color, alpha=0.6, linestyle="--", lw=0.9)
        cue_ax.annotate("", xy=(seconds, 0.95), xytext=(seconds, 0.38),
                        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.4})
        cue_ax.text(seconds, 0.2, CLASS_NAMES[int(label)], color=color,
                    ha="center", va="center", fontsize=9)

    for i, epoch in enumerate(epochs):
        erp_ax.plot(times_ms, epoch, color="#6a9cce", alpha=0.16, lw=0.6,
                    label=f"Individual tongue trials (n={len(epochs)})" if i == 0 else None)
    erp_ax.plot(times_ms, epochs.mean(axis=0), color="#152b3d", lw=2.5,
                label="Across-trial mean")
    erp_ax.axvspan(times_ms[0], 0, color="#dce6ee", alpha=0.5, label="Pre-cue baseline")
    erp_ax.axvline(0, color="#13866b", linestyle="--", lw=1.2)
    erp_ax.axhline(0, color="#b2bdc7", lw=0.8)
    erp_ax.set_title(f"C. Cue-aligned tongue trials and average | channel {channel}", loc="left", fontsize=12)
    erp_ax.set_xlabel("Time relative to tongue-imagery cue (ms)")
    erp_ax.set_ylabel("Baseline-corrected amplitude (µV)")
    erp_ax.set_xlim(times_ms[0], times_ms[-1])
    erp_ax.grid(alpha=0.15)
    erp_ax.legend(loc="upper right", frameon=False, fontsize=9)

    for ax in (raw_ax, cue_ax, erp_ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    cue_ax.spines["left"].set_visible(False)
    fig.suptitle(f"Subject 04 | From continuous EEG to a cue-locked average", fontsize=17, y=0.98)
    fig.text(0.5, 0.935, f"A04T.mat | {channel} | {fs} Hz | tongue = class 4 | "
                         "green arrows mark tongue cues", ha="center", fontsize=10, color="#596775")
    fig.text(0.5, 0.025, "Average: same channel across repeated tongue trials. "
                         "Source-flagged trials excluded; no additional filtering or re-referencing.",
             ha="center", fontsize=9, color="#596775")
    fig.subplots_adjust(top=0.875, bottom=0.1, left=0.09, right=0.97)
    return fig


def main():
    base = Path(__file__).resolve().parent
    data_path = base / "archive" / "A04T.mat"
    if not data_path.is_file():
        data_path = Path.home() / "Desktop" / "BCI_Signal_processing" / "archive" / "A04T.mat"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_path)
    parser.add_argument("--channel", choices=CHANNEL_NAMES, default="Cz")
    parser.add_argument("--output", type=Path, default=base / "outputs" / "subject04_tongue_erp.png")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    epochs, times, first_run, fs, flagged, incomplete = prepare_data(args.data, args.channel)
    print(f"Channel: {args.channel}; epochs: {epochs.shape}; sampling rate: {fs} Hz")
    print(f"Excluded: {flagged} source-flagged trials, {incomplete} incomplete/nonfinite trials")
    print("Window: -200 to +800 ms around the cue; baseline: -200 to <0 ms")
    fig = draw_figure(epochs, times, first_run, fs, args.channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    print(f"Saved: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()