from scipy.interpolate import splprep, splev
import numpy as np

class PoseBuffer:
    def __init__(self, buffer_duration=5.0):
        """
        Initializes the PoseBuffer with a specific buffer duration.
        :param buffer_duration: Duration in seconds to keep the data in the buffer
        """
        self.buffer_duration = buffer_duration  # Duration to keep past data in seconds
        self.timestamps = []  # Timestamps of the measurements
        self.positions = []  # Position measurements

    def add_measurement(self, pose, timestamp):
        """
        Add a new measurement to the buffer.
        :param pose: A tuple (x, y, z)
        :param timestamp: A float representing the time the measurement was taken
        """
        self.timestamps.append(timestamp)
        self.positions.append(pose)

        # Remove measurements that are outside the buffer duration
        current_time = self.timestamps[-1]
        self.timestamps = [t for t in self.timestamps if current_time - t <= self.buffer_duration]
        self.positions = self.positions[-len(self.timestamps):]

    def get_regularly_sampled_positions(self, dt):
        """
        Samples positions at the requested dt for the requested duration.
        :param dt: The timestep at which to sample
        :return: A list of positions sampled at interval dt
        """
        if len(self.timestamps) < 2:
            # Not enough points to create a spline
            return None

        # Normalize timestamps
        t_min = min(self.timestamps)
        t_max = max(self.timestamps)
        normalized_times = [(t - t_min) / (t_max - t_min) for t in self.timestamps]

        # Fit the spline to the positions
        positions = np.array(self.positions)
        tck, _ = splprep(positions.T, u=normalized_times)

        # Sample at requested dt
        sample_times = np.arange(0, 1, dt/(t_max - t_min))
        sampled_positions = splev(sample_times, tck)
        
        # Convert to list of tuples
        sampled_positions = list(zip(*sampled_positions))

        return sampled_positions

# # Example usage
# pose_buffer = PoseBuffer(buffer_duration=5.0)

# # Adding some measurements (assuming irregular intervals)
# pose_buffer.add_measurement((1.0, 2.0, 3.0), 0.5)
# pose_buffer.add_measurement((1.5, 2.5, 3.5), 1.5)
# pose_buffer.add_measurement((2.0, 3.0, 4.0), 2.5)
# pose_buffer.add_measurement((2.5, 3.5, 4.5), 4.0)

# # Get regularly sampled positions at 1 second intervals
# regularly_sampled_positions = pose_buffer.get_regularly_sampled_positions(1.0)
# print(regularly_sampled_positions)
