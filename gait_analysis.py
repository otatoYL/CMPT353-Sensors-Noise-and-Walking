import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime


# Ensure required directories exist
def ensure_directories():
    for dir_name in ['data', 'results']:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)


# 1. Load data
def load_data(filename):
    try:
        # Adjust for specific app output format if needed
        data = pd.read_csv(filename)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


# 2. Preprocessing and calibration
def calibrate(data, axis_name, calibration_seconds=1, sampling_rate=50):
    # Use stationary data for calibration
    start_samples = int(calibration_seconds * sampling_rate)
    end_samples = int(calibration_seconds * sampling_rate)

    if len(data) <= start_samples + end_samples:
        print("Warning: Data too short for reliable calibration")
        return data

    start_bias = data[:start_samples].mean()
    end_bias = data[-end_samples:].mean()
    avg_bias = (start_bias + end_bias) / 2

    print(f"Removing {axis_name} axis bias: {avg_bias:.4f}")

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
    # Simple threshold-based step detection with hysteresis
    steps = []
    step_count = 0

    # Hysteresis to avoid multiple triggers on noisy signal
    above_threshold = False

    for i in range(1, len(data)):
        if not above_threshold and data[i] >= threshold:
            above_threshold = True
            steps.append(i)
            step_count += 1
        elif above_threshold and data[i] < threshold * 0.7:  # Lower threshold for state change
            above_threshold = False

    return steps, step_count


# 5. Calculate gait metrics
def calculate_gait_features(steps, data, time_data):
    # Calculate cadence (steps per minute) and other metrics
    if len(steps) < 2:
        print("Warning: Not enough steps detected for reliable analysis")
        return {
            'cadence': 0,
            'step_count': len(steps),
            'avg_step_time': 0,
            'stride_regularity': 0
        }

    total_time = time_data[steps[-1]] - time_data[steps[0]]
    if total_time <= 0:
        print("Warning: Invalid time data")
        return None

    cadence = (len(steps) - 1) / (total_time / 60)

    # Calculate average time between steps
    step_times = []
    for i in range(1, len(steps)):
        step_times.append(time_data[steps[i]] - time_data[steps[i - 1]])

    avg_step_time = np.mean(step_times)

    # Stride regularity (variation in step timing)
    stride_regularity = np.std(step_times) / avg_step_time if avg_step_time > 0 else 0

    return {
        'cadence': cadence,
        'step_count': len(steps),
        'avg_step_time': avg_step_time,
        'stride_regularity': stride_regularity
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


# 7. Calculate walking speed estimate
def estimate_walking_speed(steps, time_data, height_cm=170, gender='unknown'):
    if len(steps) < 2:
        return 0, 0

    # Estimate stride length based on height
    # Rough approximation: stride length ≈ 0.415 * height for men, 0.413 * height for women
    multiplier = 0.415 if gender.lower() == 'male' else 0.413 if gender.lower() == 'female' else 0.414
    stride_length_m = (height_cm * multiplier) / 100

    # Calculate time elapsed
    total_time_s = time_data[steps[-1]] - time_data[steps[0]]
    if total_time_s <= 0:
        return 0, 0

    # Calculate distance and speed
    step_count = len(steps) - 1
    distance_m = step_count * stride_length_m / 2  # Divide by 2 because 2 steps = 1 stride
    speed_mps = distance_m / total_time_s

    return speed_mps, distance_m


# 8. Save results to file
def save_results(results, filename='gait_analysis_results.txt'):
    filepath = os.path.join('results', filename)
    with open(filepath, 'w') as f:
        f.write(f"Gait Analysis Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")

        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.2f}\n")
            else:
                f.write(f"{key}: {value}\n")

    print(f"Results saved to {filepath}")


# Main function
def process_walking_data(filename, sampling_rate=50, height_cm=170, gender='unknown', threshold=0.2):
    print(f"Processing walking data: {filename}")
    ensure_directories()

    # Load and preprocess data
    raw_data = load_data(filename)
    print(f"Loaded {len(raw_data)} samples")

    # Extract relevant columns
    if 'time' not in raw_data.columns:
        # Create time column if not provided
        raw_data['time'] = np.arange(len(raw_data)) / sampling_rate

    time = raw_data['time']

    # Check for required columns
    required_cols = ['acc_x', 'acc_y', 'acc_z']
    for col in required_cols:
        if col not in raw_data.columns:
            print(f"Error: Missing required column {col}")
            sys.exit(1)

    acc_x = raw_data['acc_x']
    acc_y = raw_data['acc_y']
    acc_z = raw_data['acc_z']

    # Calculate acceleration magnitude
    acc_mag = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)

    # Calibrate and filter
    print("\nCalibrating sensor data...")
    calibrated_x = calibrate(acc_x, 'X')
    calibrated_y = calibrate(acc_y, 'Y')
    calibrated_z = calibrate(acc_z, 'Z')
    calibrated_mag = np.sqrt(calibrated_x ** 2 + calibrated_y ** 2 + calibrated_z ** 2)

    print("\nApplying Butterworth filter...")
    filtered_x = apply_filter(calibrated_x, cutoff=2.0, fs=sampling_rate)
    filtered_y = apply_filter(calibrated_y, cutoff=2.0, fs=sampling_rate)
    filtered_z = apply_filter(calibrated_z, cutoff=2.0, fs=sampling_rate)
    filtered_mag = apply_filter(calibrated_mag, cutoff=2.0, fs=sampling_rate)

    # Detect steps (primarily using vertical acceleration)
    print("\nDetecting steps...")
    steps, step_count = detect_steps(filtered_z, threshold=threshold)
    print(f"Detected {step_count} steps")

    # Calculate gait parameters
    print("\nCalculating gait features...")
    features = calculate_gait_features(steps, filtered_z, time)
    if not features:
        print("Error calculating gait features")
        return None

    # Estimate walking speed
    speed_mps, distance_m = estimate_walking_speed(steps, time, height_cm, gender)

    # Analyze frequency components
    print("\nPerforming frequency analysis...")
    freqs, magnitude, dominant_freq = analyze_frequency(filtered_z, sampling_rate)

    # Create results file base name
    base_filename = os.path.splitext(os.path.basename(filename))[0]

    # Visualization
    print("\nGenerating visualizations...")
    plt.figure(figsize=(12, 12))

    # Plot raw and filtered data for vertical axis
    plt.subplot(4, 1, 1)
    plt.plot(time, acc_z, 'b-', alpha=0.3, label='Raw Z-axis')
    plt.plot(time, filtered_z, 'r-', label='Filtered Z-axis')
    plt.legend()
    plt.title('Vertical Acceleration Data (Z-axis)')
    plt.ylabel('Acceleration (m/s²)')

    # Plot acceleration magnitude
    plt.subplot(4, 1, 2)
    plt.plot(time, acc_mag, 'g-', alpha=0.3, label='Raw magnitude')
    plt.plot(time, filtered_mag, 'r-', label='Filtered magnitude')
    plt.legend()
    plt.title('Acceleration Magnitude')
    plt.ylabel('Acceleration (m/s²)')

    # Mark detected steps
    plt.subplot(4, 1, 3)
    plt.plot(time, filtered_z)
    if steps:
        plt.plot(time.iloc[steps], filtered_z.iloc[steps], 'go', label='Steps')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title(f'Step Detection (Count: {step_count}, Cadence: {features["cadence"]:.2f} steps/min)')
    plt.ylabel('Acceleration (m/s²)')
    plt.legend()

    # Plot frequency spectrum
    plt.subplot(4, 1, 4)
    plt.plot(freqs, magnitude)
    plt.axvline(x=dominant_freq, color='r', linestyle='--',
                label=f'Dominant freq: {dominant_freq:.2f} Hz')
    plt.title('Frequency Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('results', f'{base_filename}_analysis.png'))
    plt.show()

    # Combine all results
    results = {
        'step_count': step_count,
        'cadence': features['cadence'],
        'avg_step_time': features['avg_step_time'],
        'stride_regularity': features['stride_regularity'],
        'walking_speed_m_per_s': speed_mps,
        'estimated_distance_m': distance_m,
        'dominant_frequency_hz': dominant_freq
    }

    # Save results to file
    save_results(results, f'{base_filename}_results.txt')

    print("\nAnalysis complete!")
    return results


# Example usage
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python gait_analysis.py your_data_file.csv [sampling_rate] [height_cm] [gender]")
        print("Example: python gait_analysis.py data/walking_data.csv 50 170 male")
        sys.exit(1)

    data_file = sys.argv[1]

    # Optional parameters
    sampling_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    height_cm = float(sys.argv[3]) if len(sys.argv) > 3 else 170
    gender = sys.argv[4].lower() if len(sys.argv) > 4 else 'unknown'

    results = process_walking_data(data_file, sampling_rate, height_cm, gender)

    # Print summary to console
    if results:
        print("\nSummary Results:")
        print(f"Steps detected: {results['step_count']}")
        print(f"Cadence: {results['cadence']:.2f} steps/minute")
        print(f"Walking speed: {results['walking_speed_m_per_s']:.2f} m/s")
        print(f"Dominant frequency: {results['dominant_frequency_hz']:.2f} Hz")