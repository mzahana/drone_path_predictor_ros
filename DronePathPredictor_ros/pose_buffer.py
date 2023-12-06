#!/usr/bin/env python3
import numpy as np

class PoseBuffer:
    def __init__(self, buffer_duration=5.0, dt=0.1):
        """
        Initializes the PoseBuffer with a specific buffer duration.
        :param buffer_duration: Duration in seconds to keep the data in the buffer
        """
        self.dt = dt
        self.buffer_duration = buffer_duration  # Duration to keep past data in seconds
        self.timestamps = []  # Timestamps of the measurements
        self.positions = []  # Position measurements

        self.trajectory_to_evaluate = []
        self.trajectory_to_evaluate_timestamps = []
        self.evaluation_buffer = []
        self.evaluation_buffer_timestamps = []
        self.evaluation_time = 0.0

    def update_buffer(self, pose, timestamp):
        """
        Add a new measurement to the buffer.
        :param pose: A tuple (x, y, z)
        :param timestamp: A float representing the time the measurement was taken
        """
        if len(self.timestamps)<1:
            # print(f'len(self.timestamps): {len(self.timestamps)} < 1. Adding the first measurement')
            self.timestamps.append(timestamp)
            self.positions.append(pose)
        # print(f"len(self.positions) = {len(self.positions)}. len(self.timestampe) = {len(self.timestamps)}")

        p1=self.positions[-1]
        t1=self.timestamps[-1]
        p=self.interpolate_3d(p1, t1, pose, timestamp, self.dt)
        if p is None:
            # print("Could not interpolate")
            return None
        # print(f"interpolated, p={p}")
        
        self.timestamps.append(self.timestamps[-1]+self.dt)
        self.positions.append(p)

        N = int(self.buffer_duration/self.dt)
        if (len(self.positions) > N):
            self.timestamps = self.timestamps[-N:]
            self.positions = self.positions[-N:]

        return True

    def update_evaluation_buffer(self, pose, timestamp, N):
        # there is no traj to evaluate
        if len(self.trajectory_to_evaluate) < 1 or len(self.trajectory_to_evaluate_timestamps) < 1:
            return False
        
        self.evaluation_time = self.trajectory_to_evaluate_timestamps[0]
        if timestamp < self.evaluation_time:
            self.t0 = timestamp
            self.p0 = pose

        if len(self.evaluation_buffer) < N :    
            if timestamp > self.evaluation_time:
                interpolated_p = self.interpolate_3d_at_time(self.p0, self.t0, pose, timestamp, self.evaluation_time)
                self.evaluation_buffer.append(interpolated_p)
                self.evaluation_buffer_timestamps.append(self.evaluation_time)
                self.trajectory_to_evaluate_timestamps = self.trajectory_to_evaluate_timestamps[1:]
                return False
        else:
            return True

    def evaluate_trajectory(self):
        if len(self.evaluation_buffer) <1:
            return None
        
        if len(self.evaluation_buffer) != len(self.trajectory_to_evaluate):
            return None
        error_sum = 0
        for point_a, point_b in zip(self.trajectory_to_evaluate, self.evaluation_buffer):
            # Unpack the points
            x_a, y_a, z_a = point_a
            x_b, y_b, z_b = point_b

            # Calculate the squared distance between the points
            squared_distance = (x_a - x_b) ** 2 + (y_a - y_b) ** 2 + (z_a - z_b) ** 2

            # Add to the total error
            error_sum += squared_distance

        # Calculate the mean squared error
        mse = error_sum / len(self.evaluation_buffer) # MSE
        rmse = mse**0.5 # RMSE
        return mse, rmse
        

    def reset_evaluation_trajectory(self):
        self.evaluation_buffer =[]
        self.evaluation_buffer_timestamps = []
        self.trajectory_to_evaluate = []
        self.trajectory_to_evaluate_timestamps = []
        
    def get_regularly_sampled_positions(self):
        return self.positions.copy()
    
    def interpolate_3d(self, P1, t1, P2, t2, dt):
        """
        Interpolates to find a 3D point at a specified time step after the first point.

        Parameters:
        P1 (tuple): The (x, y, z) coordinates of the first point.
        t1 (float): The time associated with the first point.
        P2 (tuple): The (x, y, z) coordinates of the second point.
        t2 (float): The time associated with the second point.
        dt (float): The time step after t1 for which to find the new point.

        Returns:
        tuple: The (x, y, z) coordinates of the interpolated point.
        """

        # Unpack the first and second points
        x1, y1, z1 = P1
        x2, y2, z2 = P2

        # Check if dt is valid
        if dt<0:
            # print(f'dt {dt} < 0')
            return None
        if dt > (t2 - t1):
            # print(f"t1= {t1}, t2 = {t2}\n")
            # print(f"dt {dt} should be within the range of (t2 - t1): {t2-t1}")
            return None

        # Calculate the interpolation factor
        fraction = dt / (t2 - t1)

        # Calculate the interpolated coordinates
        x = x1 + (x2 - x1) * fraction
        y = y1 + (y2 - y1) * fraction
        z = z1 + (z2 - z1) * fraction

        # Return the interpolated point
        return (x, y, z)
    
    def interpolate_3d_at_time(self, P1, t1, P2, t2, t):
        """
        Interpolates to find a 3D point at a specified time between the first and second points.

        Parameters:
        P1 (tuple): The (x, y, z) coordinates of the first point.
        t1 (float): The time associated with the first point.
        P2 (tuple): The (x, y, z) coordinates of the second point.
        t2 (float): The time associated with the second point.
        t (float): The specific time for which to find the new point.

        Returns:
        tuple: The (x, y, z) coordinates of the interpolated point.
        """

        # Unpack the first and second points
        x1, y1, z1 = P1
        x2, y2, z2 = P2

        # Check if t is within the range of t1 and t2
        if t < t1 or t > t2:
            return None

        # Calculate the interpolation factor
        fraction = (t - t1) / (t2 - t1)

        # Calculate the interpolated coordinates
        x = x1 + (x2 - x1) * fraction
        y = y1 + (y2 - y1) * fraction
        z = z1 + (z2 - z1) * fraction

        # Return the interpolated point
        return (x, y, z)

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
