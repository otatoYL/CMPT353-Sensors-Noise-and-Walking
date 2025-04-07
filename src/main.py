import pandas as pd
import os
import matplotlib.pyplot as plt
from analysis_utils import estimate_sampling_rate, extract_features

positions = {
    "ankle": "data/ankle.csv",
    "hand": "data/hand.csv",
    "pocket": "data/pocket.csv"
}

os.makedirs("results/plots", exist_ok=True)
summary = []

for name, path in positions.items():
    df = pd.read_csv(path)
    time = df.iloc[:, 0].values
    acc_z = df.iloc[:, -2].values
    fs = estimate_sampling_rate(time)
    peak_freq, spm, filtered, freqs, amps = extract_features(time, acc_z, fs)
    summary.append((name, peak_freq, spm))

    # plot
    plt.figure(figsize=(10, 4))
    plt.plot(time, acc_z, label='Raw')
    plt.plot(time, filtered, label='Filtered')
    plt.title(f"{name.title()} Acc Z Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}_acc_z.png")
    plt.close()

    plt.figure(figsize=(8, 3))
    plt.plot(freqs, amps)
    plt.title(f"{name.title()} - FFT Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"results/plots/{name}_fft.png")
    plt.close()