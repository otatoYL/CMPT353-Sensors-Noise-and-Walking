import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt


# 1. Load data
def load_data(filename):
    # Adjust for specific app output format if needed
    data = pd.read_csv(filename)
    return data


# 2. Preprocessing and calibration
def calibrate(data, calibration_seconds=1):
    # Use stationary data for calibration
    start_samples = int(calibration_seconds * sampling_rate)
    end_samples = int(calibration_seconds * sampling_rate)

    start_bias = data[:start_samples].mean()
    end_bias = data[-end_samples:].mean()
    avg_bias = (start_bias + end_bias) / 2

    # Remove bias
    calibrated_data = data - avg_bias
    return calibrated_data


# 3. Apply Butterworth filter
def apply_filter(data, cutoff=2.0, fs=50, order=4):
    # Design lowpass filter (cutoff Hz)
    b, a = signal.butter(order, cutoff / (fs / 2), btype='low')

    # Apply filter
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


# 4. Step detection
def detect_steps(data, threshold=0.5):
    # Simple threshold-based step detection
    steps = []
    step_count = 0

    for i in range(1, len(data)):
        if data[i - 1] < threshold and data[i] >= threshold:
            steps.append(i)
            step_count += 1

    return steps, step_count


# 5. Calculate gait metrics
def calculate_gait_features(steps, data, time_data):
    # Calculate cadence (steps per minute)
    if len(steps) < 2:
        return None

    total_time = time_data[steps[-1]] - time_data[steps[0]]
    cadence = (len(steps) - 1) / (total_time / 60)

    return {
        'cadence': cadence,
        'step_count': len(steps)
    }


# 6. Perform FFT analysis
def analyze_frequency(data, fs):
    # Get frequency components using FFT
    n = len(data)
    fft_result = np.fft.fft(data)
    freq = np.fft.fftfreq(n, 1 / fs)

    # Only keep positive frequencies
    pos_mask = freq > 0
    freqs = freq[pos_mask]
    magnitude = np.abs(fft_result[pos_mask])

    # Find dominant frequency
    dominant_idx = np.argmax(magnitude)
    dominant_freq = freqs[dominant_idx]

    return freqs, magnitude, dominant_freq


# Main function
def process_walking_data(filename, sampling_rate=50):
    # Load and preprocess data
    raw_data = load_data(filename)

    # Extract relevant columns
    time = raw_data['time']
    acc_x = raw_data['acc_x']
    acc_y = raw_data['acc_y']
    acc_z = raw_data['acc_z']

    # Calculate acceleration magnitude
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    # Calibrate and filter
    calibrated_acc = calibrate(acc_z)  # Using vertical acceleration
    filtered_acc = apply_filter(calibrated_acc, cutoff=2.0, fs=sampling_rate)

    # Detect steps
    steps, step_count = detect_steps(filtered_acc, threshold=0.2)

    # Calculate gait parameters
    features = calculate_gait_features(steps, filtered_acc, time)

    # Analyze frequency components
    freqs, magnitude, dominant_freq = analyze_frequency(filtered_acc, sampling_rate)

    # Visualization
    plt.figure(figsize=(12, 8))

    # Plot raw and filtered data
    plt.subplot(3, 1, 1)
    plt.plot(time, acc_z, 'b-', alpha=0.3, label='Raw data')
    plt.plot(time, filtered_acc, 'r-', label='Filtered data')
    plt.legend()
    plt.title('Accelerometer Data')

    # Mark detected steps
    plt.subplot(3, 1, 2)
    plt.plot(time, filtered_acc)
    plt.plot(time[steps], filtered_acc[steps], 'go', label='Steps')
    plt.title(f'Step Detection (Count: {step_count}, Cadence: {features["cadence"]:.2f} steps/min)')
    plt.legend()

    # Plot frequency spectrum
    plt.subplot(3, 1, 3)
    plt.plot(freqs, magnitude)
    plt.axvline(x=dominant_freq, color='r', linestyle='--',
                label=f'Dominant freq: {dominant_freq:.2f} Hz')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.legend()

    plt.tight_layout()
    plt.show()

    return {
        'step_count': step_count,
        'cadence': features['cadence'],
        'dominant_frequency': dominant_freq
    }


# Example usage
if __name__ == "__main__":
    sampling_rate = 50  # Hz
    results = process_walking_data('walking_data.csv', sampling_rate)
    print(f"Steps detected: {results['step_count']}")
    print(f"Cadence: {results['cadence']:.2f} steps/minute")
    print(f"Dominant frequency: {results['dominant_frequency']:.2f} Hz")