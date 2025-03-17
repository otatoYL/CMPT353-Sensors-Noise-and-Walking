import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import os


class SensorData:
    """Class for handling smartphone sensor data collection and preprocessing"""

    def __init__(self, sampling_rate=50):
        """Initialize sensor data object with given sampling rate"""
        self.sampling_rate = sampling_rate
        self.calibration_data = None
        self.data = None
        self.orientation_data = None

    def load_from_file(self, filename):
        """Load sensor data from CSV file"""
        try:
            data = pd.read_csv(filename)
            required_cols = ['acc_x', 'acc_y', 'acc_z']

            # Check if required columns exist
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing required columns: {missing_cols}")
                return None

            # If time column doesn't exist, create it
            if 'time' not in data.columns:
                data['time'] = np.arange(len(data)) / self.sampling_rate

            self.data = data
            print(f"Loaded {len(data)} samples from {filename}")
            return data

        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def collect_calibration_data(self, seconds=5):
        """
        This method would integrate with a sensor collection app
        For now, it's a placeholder to show how calibration data could be collected
        """
        print(f"Stand still for {seconds} seconds to calibrate sensors...")
        # In a real implementation, this would interface with a data collection app
        print("Calibration complete")

    def apply_calibration(self, start_seconds=1, end_seconds=1):
        """Apply calibration to remove sensor bias"""
        if self.data is None:
            print("No data loaded")
            return None

        calibrated_data = self.data.copy()

        # Calculate samples for calibration periods
        start_samples = int(start_seconds * self.sampling_rate)
        end_samples = int(end_seconds * self.sampling_rate)

        if len(self.data) < start_samples + end_samples:
            print("Warning: Data too short for reliable calibration")
            return calibrated_data

        # Calculate bias for each axis using specified periods
        for axis in ['acc_x', 'acc_y', 'acc_z']:
            if axis in self.data.columns:
                start_bias = self.data[axis][:start_samples].mean()
                end_bias = self.data[axis][-end_samples:].mean()
                avg_bias = (start_bias + end_bias) / 2

                print(f"Removing {axis} bias: {avg_bias:.4f}")
                calibrated_data[axis] = self.data[axis] - avg_bias

        return calibrated_data

    def calculate_magnitude(self, data=None):
        """Calculate acceleration magnitude from XYZ components"""
        if data is None:
            data = self.data

        if data is None:
            print("No data available")
            return None

        # Check if required columns exist
        required_cols = ['acc_x', 'acc_y', 'acc_z']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Missing required columns for magnitude calculation: {missing_cols}")
            return None

        # Calculate magnitude
        data['acc_magnitude'] = np.sqrt(
            data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2
        )

        return data

    def correct_orientation(self, data=None, gyro_cols=None):
        """
        Advanced feature: Correct sensor orientation using gyroscope data
        This helps maintain consistent "forward" and "down" directions
        """
        if data is None:
            data = self.data

        if data is None:
            print("No data available")
            return None

        # Check if gyroscope data is available
        if gyro_cols is None:
            gyro_cols = ['gyro_x', 'gyro_y', 'gyro_z']

        missing_gyro = [col for col in gyro_cols if col not in data.columns]
        if missing_gyro:
            print(f"Warning: Missing gyroscope data columns: {missing_gyro}")
            print("Orientation correction skipped.")
            return data

        corrected_data = data.copy()
        dt = 1.0 / self.sampling_rate

        # Initial orientation (identity quaternion)
        orientation = Rotation.identity()

        # Store corrected accelerations
        acc_corrected_x = []
        acc_corrected_y = []
        acc_corrected_z = []

        # For each time step
        for i in range(len(data)):
            # Get angular velocities
            omega_x = data[gyro_cols[0]].iloc[i]
            omega_y = data[gyro_cols[1]].iloc[i]
            omega_z = data[gyro_cols[2]].iloc[i]

            # Update orientation using gyroscope data
            omega = np.array([omega_x, omega_y, omega_z])
            delta_q = Rotation.from_rotvec(omega * dt)
            orientation = orientation * delta_q

            # Get original acceleration
            acc = np.array([
                data['acc_x'].iloc[i],
                data['acc_y'].iloc[i],
                data['acc_z'].iloc[i]
            ])

            # Transform acceleration to global frame
            acc_global = orientation.apply(acc)

            # Store corrected values
            acc_corrected_x.append(acc_global[0])
            acc_corrected_y.append(acc_global[1])
            acc_corrected_z.append(acc_global[2])

        # Update data with corrected values
        corrected_data['acc_x_corrected'] = acc_corrected_x
        corrected_data['acc_y_corrected'] = acc_corrected_y
        corrected_data['acc_z_corrected'] = acc_corrected_z

        return corrected_data

    def visualize_raw_data(self, filename=None):
        """Visualize raw sensor data"""
        if self.data is None:
            print("No data loaded")
            return

        plt.figure(figsize=(12, 8))

        # Plot acceleration data
        plt.subplot(2, 1, 1)
        plt.plot(self.data['time'], self.data['acc_x'], 'r-', label='X-axis')
        plt.plot(self.data['time'], self.data['acc_y'], 'g-', label='Y-axis')
        plt.plot(self.data['time'], self.data['acc_z'], 'b-', label='Z-axis')

        plt.title('Raw Accelerometer Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.legend()
        plt.grid(True)

        # Plot magnitude if available
        if 'acc_magnitude' in self.data.columns:
            plt.subplot(2, 1, 2)
            plt.plot(self.data['time'], self.data['acc_magnitude'], 'k-', label='Magnitude')
            plt.title('Acceleration Magnitude')
            plt.xlabel('Time (s)')
            plt.ylabel('Acceleration (m/s²)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()

        # Save figure if filename provided
        if filename:
            if not os.path.exists('results'):
                os.makedirs('results')
            plt.savefig(os.path.join('results', filename))

        plt.show()


def record_sensor_data(output_file, recording_time=30, sampling_rate=50):
    """
    Placeholder function for recording sensor data
    In a real implementation, this would interface with a smartphone app
    """
    print(f"Recording sensor data for {recording_time} seconds at {sampling_rate} Hz...")
    print(f"Data will be saved to {output_file}")

    print("\nThis is a placeholder function. To collect real data:")
    print("1. Use an app like Physics Toolbox Sensor Suite on your smartphone")
    print("2. Record acceleration data and export it as CSV")
    print("3. Make sure the CSV has columns: time, acc_x, acc_y, acc_z")
    print("4. Stand still for 1-2 seconds at the beginning and end for calibration")

    return None


def example_usage():
    """Example usage of the SensorData class"""
    # Create sensor data object
    sensor = SensorData(sampling_rate=50)

    # Load data
    data = sensor.load_from_file('data/walking_data.csv')

    # Apply calibration
    calibrated_data = sensor.apply_calibration()

    # Calculate magnitude
    data_with_magnitude = sensor.calculate_magnitude(calibrated_data)

    # Visualize raw data
    sensor.visualize_raw_data('raw_data_visualization.png')

    return data_with_magnitude


if __name__ == "__main__":
    print("Sensor data module. Import this module to use its functions.")
    print("Example usage:")
    print("from sensor import SensorData")
    print("sensor = SensorData()")
    print("data = sensor.load_from_file('your_data.csv')")