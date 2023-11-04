#!/usr/bin/env python3

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
