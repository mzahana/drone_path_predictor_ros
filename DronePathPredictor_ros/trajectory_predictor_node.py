#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from .trajectory_predictor import Predictor
from .pose_buffer import PoseBuffer
import numpy as np

class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__('trajectory_predictor_node')

        # Declare and get parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('position_model_path', 'path/to/your/position_model.pt'),
                ('velocity_model_path', 'path/to/your/velocity_model.pt'),
                ('position_npz_path', 'path/to/your/position_normalization_parameters.npz'),
                ('velocity_npz_path', 'path/to/your/velocity_normalization_parameters.npz'),
                ('buffer_duration', 2.0),
                ('dt', 0.1),
            ]
        )
        
        position_model_path = self.get_parameter('position_model_path').get_parameter_value().string_value
        velocity_model_path = self.get_parameter('velocity_model_path').get_parameter_value().string_value
        position_npz_path = self.get_parameter('position_npz_path').get_parameter_value().string_value
        velocity_npz_path = self.get_parameter('velocity_npz_path').get_parameter_value().string_value
        self.buffer_duration = self.get_parameter('buffer_duration').get_parameter_value().double_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value

        # Initialize the Predictor
        self.predictor = Predictor(position_model_path,
                                   velocity_model_path,
                                   position_npz_path,
                                   velocity_npz_path)
        self.get_logger().info('Initialized position and velocity models')
        
        # Create the PoseBuffer
        self.pose_buffer = PoseBuffer(buffer_duration=self.buffer_duration, dt=self.dt)
        
        # Create a subscription to the PoseArray topic
        self.create_subscription(
            PoseArray,
            'in/pose_array',  # Update this to the actual topic name
            self.pose_callback,
            10)
        # Create a publisher for the predicted path
        self.path_publisher = self.create_publisher(Path, 'out/gru_predicted_path', 10)
        self.history_path_publisher = self.create_publisher(Path, 'out/gru_history_path', 10)


    def pose_callback(self, msg: PoseArray):
        # Extract the first pose from the PoseArray
        first_pose = msg.poses[0]
        
        # Convert the pose to a tuple (x, y, z)
        position = (first_pose.position.x, first_pose.position.y, first_pose.position.z)
        
        # Add the measurement to the PoseBuffer with the current time
        t=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ret = self.pose_buffer.update_buffer(position, t)
        # if ret is None:
        #     # self.get_logger().info('update_buffer retuirned None')    
        #     return

        # Get regularly sampled positions from the PoseBuffer
        regularly_sampled_positions = self.pose_buffer.get_regularly_sampled_positions()
        # self.get_logger().info(f'# regularly_sampled_positions \n {len(regularly_sampled_positions)}')
        
        if len(regularly_sampled_positions)==int(self.buffer_duration/self.dt):
            # Duplicate the last position.
            # This is needed to compute input velocity sequence
            last_position = regularly_sampled_positions[-1]
            regularly_sampled_positions.append(last_position)
            # Convert to numpy array
            np_positions = np.array(regularly_sampled_positions)
            # print("np_position :\n", np_positions)
            
            # Predict positions from the sampled buffer data
            predicted_positions = self.predictor.predict_positions(np_positions.copy())

            predicted_positions_from_velocity = self.predictor.predict_positions_from_velocity(np_positions.copy(), self.dt)
            # print("predicted_positions_from_velocity shape: ", predicted_positions_from_velocity.shape)
            # self.get_logger().info('Got predicted_positions_from_velocity')
            
            # self.get_logger().info('Predicted Positions: {}'.format(predicted_positions))

            if predicted_positions is not None:
                # Create a Path message
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = msg.header.frame_id

                # Fill the Path message with the predicted positions
                for position in predicted_positions:
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = self.get_clock().now().to_msg()
                    pose_stamped.header.frame_id = msg.header.frame_id
                    pose_stamped.pose.position.x = position[0]
                    pose_stamped.pose.position.y = position[1]
                    pose_stamped.pose.position.z = position[2]
                    # Assume no orientation information is available; quaternion set to identity
                    pose_stamped.pose.orientation.w = 1.0
                    path_msg.poses.append(pose_stamped)

                # Publish the predicted Path message
                self.path_publisher.publish(path_msg)
                
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = msg.header.frame_id

                # Fill the Path message with the predicted positions
                for position in regularly_sampled_positions[-int(self.buffer_duration/self.dt):]:
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = self.get_clock().now().to_msg()
                    pose_stamped.header.frame_id = msg.header.frame_id
                    pose_stamped.pose.position.x = position[0]
                    pose_stamped.pose.position.y = position[1]
                    pose_stamped.pose.position.z = position[2]
                    # Assume no orientation information is available; quaternion set to identity
                    pose_stamped.pose.orientation.w = 1.0
                    path_msg.poses.append(pose_stamped)
                self.history_path_publisher.publish(path_msg)


def main(args=None):
    rclpy.init(args=args)

    trajectory_predictor_node = TrajectoryPredictorNode()

    rclpy.spin(trajectory_predictor_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically when the garbage collector destroys the node object)
    trajectory_predictor_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
