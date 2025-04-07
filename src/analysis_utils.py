import numpy as np
from scipy.fft import fft, fftfreq
from filter_utils import butterworth_filter

def estimate_sampling_rate(time_array):
    time_diffs = np.diff(time_array)
    return 1 / np.mean(time_diffs)

def extract_features(time, acc_z, fs):
    filtered = butterworth_filter(acc_z, cutoff=3.0, fs=fs, order=4)
    N = len(filtered)
    yf = fft(filtered)
    xf = fftfreq(N, 1/fs)
    pos_freqs = xf[:N // 2]
    pos_amps = np.abs(yf[:N // 2])
    peak_idx = np.argmax(pos_amps[1:]) + 1
    peak_freq = pos_freqs[peak_idx]
    steps_per_min = peak_freq * 60
    return peak_freq, steps_per_min, filtered, pos_freqs, pos_amps
