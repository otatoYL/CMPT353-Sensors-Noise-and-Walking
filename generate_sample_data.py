import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def generate_walking_data(filename='walking_data.csv', duration=10, sampling_rate=50,
                          step_frequency=1.8, noise_level=0.05):
    """
    Generate sample walking data to test the gait analysis system

    Parameters:
    - filename: Where to save the CSV
    - duration: Duration in seconds
    - sampling_rate: Samples per second
    - step_frequency: Steps per second (typical walking ~1.8 Hz)
    - noise_level: Amount of noise to add
    """
    # Create time array
    time = np.arange(0, duration, 1 / sampling_rate)
    n_samples = len(time)

    # Create base signals with calibration periods (standing still)
    calibration_period = int(sampling_rate)  # 1 second calibration at start/end

    # Generate vertical acceleration (Z axis) with walking pattern
    # Standing still for first and last second
    acc_z = np.zeros(n_samples)

    # Walking period (between calibration periods)
    walking_start = calibration_period
    walking_end = n_samples - calibration_period
    walking_time = time[walking_start:walking_end] - time[walking_start]

    # Create a realistic walking signal with steps
    # Each step has a characteristic pattern: impact, brake, and push-off
    for i in range(walking_start, walking_end):
        t = time[i] - time[walking_start]
        # Base oscillation at walking frequency
        base_signal = np.sin(2 * np.pi * step_frequency * t)

        # Add harmonics to make it more realistic
        harmonics = 0.5 * np.sin(4 * np.pi * step_frequency * t) + 0.3 * np.sin(6 * np.pi * step_frequency * t)

        # Combine signals with non-linear transformation to create sharper peaks
        acc_z[i] = np.tanh(2 * (base_signal + harmonics)) * 0.8

    # Generate X and Y accelerations (less pronounced than Z)
    acc_x = np.zeros(n_samples)
    acc_y = np.zeros(n_samples)

    # Add some smaller side-to-side motion in X axis during walking period
    for i in range(walking_start, walking_end):
        t = time[i] - time[walking_start]
        acc_x[i] = 0.3 * np.sin(np.pi * step_frequency * t)
        acc_y[i] = 0.2 * np.sin(np.pi * step_frequency * t + np.pi / 4)

    # Add noise to all signals
    acc_x += np.random.normal(0, noise_level, n_samples)
    acc_y += np.random.normal(0, noise_level, n_samples)
    acc_z += np.random.normal(0, noise_level, n_samples)

    # Add bias to simulate uncalibrated sensor
    acc_x += 0