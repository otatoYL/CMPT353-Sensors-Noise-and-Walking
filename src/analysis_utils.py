import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from bw_filter_data import butterworth_filter

def estimate_sampling_rate(time_array):
    time_diffs = np.diff(time_array)
    return 1 / np.mean(time_diffs)

# Extract peak frequency, steps/min, filtered signal, and spectrum
def extract_features(time, acc_z, fs):
    filtered = butterworth_filter(acc_z, cutoff=3.0, fs=fs, order=4)
    N = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(N, 1/fs)
    pos_freqs = xf[:N // 2]
    pos_amps = np.abs(yf[:N // 2])

    # Find peak frequency index (skipping the DC component)
    peak_idx = np.argmax(pos_amps[1:]) + 1
    peak_freq = pos_freqs[peak_idx]
    steps_per_min = peak_freq * 60

    return peak_freq, steps_per_min, filtered, pos_freqs, pos_amps

# Analyze a single CSV file: filtering, feature extraction, plotting
def analyze(file_path, name):
    df = pd.read_csv(file_path)
    if not {'seconds', 'z'}.issubset(df.columns):
        print(f"Skipping {name}:lack of necessary columns")
        return None

    t = df['seconds']
    z = df['z'].values
    fs = estimate_sampling_rate(t)
    dt = 1 / fs

    peak_freq, spm, z_f, freqs, amps = extract_features(t, z, fs)
    energy = np.sum(z_f ** 2)
    velocity = np.cumsum(z_f) * dt

    os.makedirs('results/plots', exist_ok=True)

    plt.figure()
    plt.plot(t, z, label='Raw Z')
    plt.plot(t, z_f, label='Filtered Z')
    plt.xlabel('Time (s)')
    plt.ylabel('Acceleration (m/s²)')
    plt.title(f"{name} Acceleration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'results/plots/{name}_z_time.png')
    plt.close()

    # Plot: FFT
    plt.figure()
    plt.plot(freqs, amps)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f"{name} FFT Spectrum")
    plt.tight_layout()
    plt.savefig(f'results/plots/{name}_fft.png')
    plt.close()

    # Plot: Integrated Velocity
    plt.figure()
    plt.plot(t, velocity)
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.title(f"{name} Integrated Velocity")
    plt.tight_layout()
    plt.savefig(f'results/plots/{name}_velocity.png')
    plt.close()

    print(f"{name}: {spm:.2f} steps/min, peak @ {peak_freq:.2f} Hz, energy = {energy:.2f}")

    return {
        'position': name,
        'peak_freq': round(peak_freq, 3),
        'steps_per_min': round(spm, 2),
        'energy_z': round(energy, 2)
    }

def main():
    summary = []
    os.makedirs('results', exist_ok=True)

    for fname in os.listdir('processed_data'):
        if fname.endswith('_bw.csv'):
            name = fname.replace('_bw.csv', '')
            path = os.path.join('processed_data', fname)
            try:
                result = analyze(path, name)
                if result:
                    summary.append(result)
            except Exception as e:
                print(f" Fail to analyze {fname}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv("results/summary.csv", index=False)
        print("summary.csv saved successfully")
    else:
        print("️There is no result")

if __name__ == '__main__':
    main()


