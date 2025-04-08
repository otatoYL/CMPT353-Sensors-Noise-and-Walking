import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os

def butterworth_filter(data, cutoff=3.0, fs=50.0, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def main():
    for file_name in os.listdir('processed_data'):
        if file_name.endswith('.csv') and not file_name.endswith('_bw.csv'):
            df = pd.read_csv(f'processed_data/{file_name}')
            for axis in ['x', 'y', 'z']:
                df[axis] = butterworth_filter(df[axis])

            bw_file = f'processed_data/{file_name.replace(".csv", "_bw.csv")}'
            df.to_csv(bw_file, index=False)

if __name__ == '__main__':
    main()