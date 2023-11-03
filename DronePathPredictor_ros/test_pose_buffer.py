import unittest
from pose_buffer import PoseBuffer

class TestPoseBuffer(unittest.TestCase):

    def test_linear_trajectory_interpolation(self):
        # Create an instance of the PoseBuffer
        buffer = PoseBuffer(buffer_duration=10.0)

        # Simulate a linear trajectory with a constant velocity
        start_time = 0.0
        velocity = 3.0  # Velocity in units/second
        duration = 4.0  # Duration in seconds
        dt = 0.1  # Time step in seconds

        # Add measurements to the PoseBuffer simulating the linear trajectory
        num_measurements = int(duration / dt)
        for i in range(num_measurements):
            time_stamp = start_time + i * dt
            position = (velocity * time_stamp,
                        velocity * time_stamp,
                        velocity * time_stamp)
            buffer.add_measurement(position, time_stamp)

        # Get regularly sampled positions at dt intervals
        regularly_sampled_positions = buffer.get_regularly_sampled_positions(dt)

        # Check that the interpolated data matches the expected linear trajectory
        for idx, sampled_position in enumerate(regularly_sampled_positions):
            expected_time = idx * dt
            expected_position = (velocity * expected_time,
                                 velocity * expected_time,
                                 velocity * expected_time)
            self.assertAlmostEqual(sampled_position[0], expected_position[0], places=5)
            self.assertAlmostEqual(sampled_position[1], expected_position[1], places=5)
            self.assertAlmostEqual(sampled_position[2], expected_position[2], places=5)

if __name__ == '__main__':
    unittest.main()
