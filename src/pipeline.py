import pandas as pd
import numpy as np
from scipy.fft import fft

def extract_vector_features(path, label):
    df = pd.read_csv(path)
    z = df['z'].values
    fs = 1 / np.mean(np.diff(df['seconds'].values))
    z_f = z if len(z) < 4 else z.copy()

    vector = np.zeros(300)
    length = min(len(z_f), 300)
    vector[:length] = z_f[:length]

    yf = np.abs(fft(z_f))
    dt = 1 / fs
    xf = np.fft.fftfreq(len(z_f), dt)[:len(yf)//2]
    peak_freq = xf[np.argmax(yf[1:len(yf)//2])] if len(yf) > 2 else 0

    stats = {
        "mean": np.mean(z_f),
        "std": np.std(z_f),
        "energy": np.sum(z_f ** 2),
        "peak_freq": peak_freq,
        "label": label
    }

    return vector, stats
