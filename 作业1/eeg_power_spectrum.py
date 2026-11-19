"""Plot the power spectrum of subject 04's tongue motor-imagery EEG.

Run this file in PyCharm. Dependencies: numpy, scipy, matplotlib.
The script computes a power spectrum (uV^2) for each 4-second trial,
then averages the spectra. Welch uses scaling="spectrum", not "density".
It does not compute the spectrum of the across-trial averaged waveform.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch


CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3", "C1", "Cz", "C2",
    "C4", "C6", "CP3", "CP1", "CPz", "CP2", "CP4", "P1", "Pz", "P2", "POz",
]
CLASS_ID = 4                 # 1 = left hand; 2 = right hand; 3 = feet; 4 = tongue
DURATION_SECONDS = 4.0
CUE_DELAY_SECONDS = 2.0     # Cue onset is 2 seconds after trial start.
BANDS = [
    ("Delta", 0.5, 4, "#b7c8e6"),
    ("Theta", 4, 8, "#a5d5d0"),
    ("Alpha", 8, 13, "#efd36d"),
    ("Beta", 13, 30, "#a8c5e8"),
    ("Gamma*", 30, 40, "#c9b1da"),
]


def load_tongue_trials(path, channel):
    """Extract finite, source-unflagged trials; return trials x samples in uV."""
    runs = np.atleast_1d(loadmat(
        path, struct_as_record=False, squeeze_me=True, variable_names=["data"]
    )["data"])
    channel_index = CHANNEL_NAMES.index(channel)
    trials = []
    fs = None
    flagged = 0
    incomplete = 0
    for run in runs:
        labels = np.asarray(run.y).reshape(-1).astype(int)
        if not labels.size:  # Calibration runs have no task labels.
            continue
        run_fs = int(run.fs)
        if fs is not None and fs != run_fs:
            raise ValueError("The runs have different sampling rates.")
        fs = run_fs
        starts = np.asarray(run.trial).reshape(-1).astype(int)
        flags = np.asarray(run.artifacts).reshape(-1)
        for trial_index in np.flatnonzero(labels == CLASS_ID):
            if flags.size and flags[trial_index] != 0:
                flagged += 1
                continue
            # Convert MATLAB's 1-based index, then move to cue onset.
            start = starts[trial_index] - 1 + round(CUE_DELAY_SECONDS * fs)
            stop = start + round(DURATION_SECONDS * fs)
            if start < 0 or stop > run.X.shape[0]:
                incomplete += 1
                continue
            signal = np.array(run.X[start:stop, channel_index], dtype=float, copy=True)
            if not np.isfinite(signal).all():
                incomplete += 1
                continue
            trials.append(signal)
    if not trials:
        raise ValueError("No complete, unflagged tongue trials were found.")
    return np.stack(trials), fs, flagged, incomplete


def compute_power_spectrum(trials, fs):
    # Welch: 2-second Hann windows with 50% overlap, 0.5-Hz frequency spacing.
    window_size = min(round(2 * fs), trials.shape[1])
    frequencies, trial_spectra = welch(
        trials, fs=fs, window="hann", nperseg=window_size,
        noverlap=window_size // 2, nfft=window_size,
        detrend="constant", scaling="spectrum", axis=-1,
    )
    # Average power, preserving activity whose phase varies between trials.
    # Spectrum scaling normalizes by the squared window sum, giving uV^2.
    # Its ratio to density scaling is the window's equivalent noise bandwidth
    # (0.75 Hz for these 2-second Hann windows), not just the FFT bin spacing.
    return frequencies, trial_spectra.mean(axis=0)


def draw_figure(trials, fs, frequencies, mean_spectrum, channel):
    fig, spectrum_ax = plt.subplots(figsize=(12, 6.5))
    mask = (frequencies >= 0.5) & (frequencies <= 40)
    f, power = frequencies[mask], mean_spectrum[mask]
    for name, low, high, color in BANDS:
        spectrum_ax.axvspan(low, high, color=color, alpha=0.12, zorder=0)
        band = (f >= low) & (f <= high)
        spectrum_ax.fill_between(f, 0, power, where=band, color=color, alpha=0.9, linewidth=0)
        spectrum_ax.text(
            (low + high) / 2, 0.96, f"{name}\n{low:g}–{high:g} Hz",
            transform=spectrum_ax.get_xaxis_transform(), ha="center", va="top", fontsize=10,
        )
    spectrum_ax.plot(f, power, color="#263b50", lw=1.3)
    spectrum_ax.set_xlim(0, 40)
    spectrum_ax.set_ylim(0, float(power.max()) * 1.25)
    spectrum_ax.set_xticks(np.arange(0, 41, 5))
    spectrum_ax.set_xlabel("Frequency (Hz)")
    spectrum_ax.set_ylabel("Power (µV²)")
    spectrum_ax.set_title(
        f"Mean of {len(trials)} individual-trial power spectra | Welch method",
        loc="left", fontsize=11, pad=12,
    )
    spectrum_ax.grid(axis="y", alpha=0.2)
    spectrum_ax.spines["top"].set_visible(False)
    spectrum_ax.spines["right"].set_visible(False)
    fig.suptitle(f"Subject 04 | Tongue motor imagery | Channel {channel}", fontsize=17, y=0.97)
    fig.text(0.5, 0.9, f"A04T.mat | {fs} Hz | 4 s/trial | "
             "2 s Hann windows | 50% overlap | 0.5 Hz frequency spacing",
             ha="center", fontsize=10, color="#596775")
    fig.text(0.5, 0.025, "Source-flagged trials excluded. No additional band-pass filtering. "
             "* Only the 30–40 Hz portion of gamma is displayed.",
             ha="center", fontsize=9, color="#596775")
    fig.subplots_adjust(top=0.81, bottom=0.15, left=0.1, right=0.97)
    return fig


def main():
    base = Path(__file__).resolve().parent
    data_file = base / "archive" / "A04T.mat"
    if not data_file.is_file():
        data_file = Path.home() / "Desktop" / "BCI_Signal_processing" / "archive" / "A04T.mat"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=data_file)
    parser.add_argument("--channel", choices=CHANNEL_NAMES, default="Cz")
    parser.add_argument("--output", type=Path, default=base / "outputs" / "subject04_tongue_psd.png")
    parser.add_argument("--no-show", action="store_true", help="Save the plot without a GUI window")
    args = parser.parse_args()
    print(f"Running script: {Path(__file__).resolve()}")
    trials, fs, flagged, incomplete = load_tongue_trials(args.data, args.channel)
    f, power = compute_power_spectrum(trials, fs)
    print(f"Subject 04 | tongue | {args.channel} | {len(trials)} trials | {fs} Hz")
    print(f"Trial matrix: {trials.shape}; excluded {flagged} flagged and {incomplete} incomplete trials")
    print("Welch power spectrum (scaling='spectrum') computed separately for each trial, then averaged; units: uV^2")
    fig = draw_figure(trials, fs, f, power, args.channel)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    print(f"Saved plot: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
