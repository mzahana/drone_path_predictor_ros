#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from .trajectory_predictor import Predictor
from .pose_buffer import PoseBuffer
import numpy as np
import time

class TrajectoryPredictorNode(Node):
    def __init__(self):
        super().__init__('trajectory_predictor_node')

        # Declare and get parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('position_model_path', 'path/to/your/position_model.pt'),
                ('velocity_model_path', 'path/to/your/velocity_model.pt'),
                ('position_stats_file', 'position_stats_file'),
                ('velocity_stats_file', 'velocity_stats_file'),
                ('buffer_duration', 2.0),
                ('dt', 0.1),
                ('pos_hidden_dim', 64),
                ('pos_num_layers', 2),
                ('pos_dropout', 0.5),
                ('vel_hidden_dim', 64),
                ('vel_num_layers', 2),
                ('vel_dropout', 0.5),
                ('use_velocity_prediction', False),
                ('use_whitening', False)
            ]
        )
        
        position_model_path = self.get_parameter('position_model_path').get_parameter_value().string_value
        velocity_model_path = self.get_parameter('velocity_model_path').get_parameter_value().string_value
        position_stats_file = self.get_parameter('position_stats_file').get_parameter_value().string_value
        velocity_stats_file = self.get_parameter('velocity_stats_file').get_parameter_value().string_value
        self.buffer_duration = self.get_parameter('buffer_duration').get_parameter_value().double_value
        self.dt = self.get_parameter('dt').get_parameter_value().double_value
        self.pos_hidden_dim = self.get_parameter('pos_hidden_dim').get_parameter_value().integer_value
        self.pos_num_layers = self.get_parameter('pos_num_layers').get_parameter_value().integer_value
        self.pos_dropout = self.get_parameter('pos_dropout').get_parameter_value().double_value
        self.vel_hidden_dim = self.get_parameter('vel_hidden_dim').get_parameter_value().integer_value
        self.vel_num_layers = self.get_parameter('vel_num_layers').get_parameter_value().integer_value
        self.vel_dropout = self.get_parameter('vel_dropout').get_parameter_value().double_value
        self.use_velocity_prediction = self.get_parameter('use_velocity_prediction').get_parameter_value().bool_value
        self.use_whitening = self.get_parameter('use_whitening').get_parameter_value().bool_value

        # Initialize the Predictor
        self.predictor = Predictor(position_model_path,
                                   velocity_model_path,
                                   position_stats_file,
                                   velocity_stats_file,
                                   pos_hidden_dim=self.pos_hidden_dim, pos_num_layers=self.pos_num_layers, pos_dropout=self.pos_dropout,
                                   vel_hidden_dim=self.vel_hidden_dim, vel_num_layers=self.vel_num_layers, vel_dropout=self.vel_dropout,
                                   use_whitening= self.use_whitening)
        self.get_logger().info('Initialized position and velocity models')
        
        # Create the PoseBuffer
        self.pose_buffer = PoseBuffer(buffer_duration=self.buffer_duration, dt=self.dt)

        self.mse_sum = 0.0
        self.rmse_sum = 0.0
        self.evaluation_counter = 0
        self.max_counter = 100

        self.execution_timer_counter_ = 0.0
        self.aggregate_execution_time_ = 0.0
        
        # Create a subscription to the PoseArray topic
        self.create_subscription(
            PoseArray,
            'in/pose_array',  # Update this to the actual topic name
            self.pose_callback,
            10)
        # Create a publisher for the predicted path
        self.path_publisher = self.create_publisher(Path, 'out/gru_predicted_path', 10)
        self.history_path_publisher = self.create_publisher(Path, 'out/gru_history_path', 10)
        self.actual_predicted_path_pub = self.create_publisher(Path, 'out/actual_predicted_path', 10)
        self.evaluated_predicted_path_pub = self.create_publisher(Path, 'out/evaluated_predicted_path', 10)
        self.evaluation_mse_pub = self.create_publisher(Float32, 'out/evaluation_mse', 10)
        self.evaluation_rmse_pub = self.create_publisher(Float32, 'out/evaluation_rmse', 10)


    def pose_callback(self, msg: PoseArray):
        # Compute execution time
        t0 = time.time()
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
        # Starting point: the last timestamp in the existing list plus dt
        start_timestamp = self.pose_buffer.timestamps[-1] + self.dt

        # Generate the new list of timestamps
        future_timestamps = [start_timestamp + i * self.dt for i in range(len(regularly_sampled_positions))]

        # self.get_logger().info(f'# regularly_sampled_positions \n {len(regularly_sampled_positions)}')
        
        if len(regularly_sampled_positions)==int(self.buffer_duration/self.dt):
            # Duplicate the last position.
            # This is needed to compute input velocity sequence
            # last_position = regularly_sampled_positions[-1]
            # regularly_sampled_positions.append(last_position)
            # Convert to numpy array
            np_positions = np.array(regularly_sampled_positions)
            # print("Shape of nps_positions", np_positions.shape)
            # print("np_position :\n", np_positions)
            
            # Predict positions from the sampled buffer data
            predicted_positions = self.predictor.predict_positions(np_positions.copy())

            predicted_positions_from_velocity = self.predictor.predict_positions_from_velocity(np_positions.copy(), self.dt)
            # print("predicted_positions_from_velocity shape: ", predicted_positions_from_velocity.shape)
            # self.get_logger().info('Got predicted_positions_from_velocity')
            
            # self.get_logger().info('Predicted Positions: {}'.format(predicted_positions))

            if self.use_velocity_prediction:
                predictions = predicted_positions_from_velocity
            else:
                predictions = predicted_positions

            if predictions is not None:

                if len(self.pose_buffer.trajectory_to_evaluate) < 1:
                    self.pose_buffer.trajectory_to_evaluate = predictions.tolist()
                    self.pose_buffer.trajectory_to_evaluate_timestamps = future_timestamps.copy()

                if(self.pose_buffer.update_evaluation_buffer(position, t, len(predictions))):
                    mse, rmse = self.pose_buffer.evaluate_trajectory()
                    
                    # Compute Average MSE, and average RMSE for a maximum number of points =  self.max_counter
                    if self.evaluation_counter <= self.max_counter:
                        self.mse_sum += mse
                        self.rmse_sum += rmse
                        self.evaluation_counter +=1

                        avg_mse = self.mse_sum / self.evaluation_counter
                        avg_rmse = self.rmse_sum / self.evaluation_counter
                        self.get_logger().info(f'Average MSE: {avg_mse}')
                        self.get_logger().info(f'Average RMSE: {avg_rmse}')
                        self.get_logger().info(f'evaluation_counter: {self.evaluation_counter}')

                        float_msg = Float32()
                        float_msg.data = mse
                        self.evaluation_mse_pub.publish(float_msg)
                        float_msg.data = rmse
                        self.evaluation_rmse_pub.publish(float_msg)
                    
                    # Create a Path message for predictions
                    path_msg = Path()
                    path_msg.header.stamp = self.get_clock().now().to_msg()
                    path_msg.header.frame_id = msg.header.frame_id
                    # Fill the Path message with the predicted positions
                    for position in self.pose_buffer.trajectory_to_evaluate:
                        pose_stamped = PoseStamped()
                        pose_stamped.header.stamp = self.get_clock().now().to_msg()
                        pose_stamped.header.frame_id = msg.header.frame_id
                        pose_stamped.pose.position.x = float(position[0])
                        pose_stamped.pose.position.y = float(position[1])
                        pose_stamped.pose.position.z = float(position[2])
                        # Assume no orientation information is available; quaternion set to identity
                        pose_stamped.pose.orientation.w = 1.0
                        path_msg.poses.append(pose_stamped)

                    # Publish the predicted Path message
                    self.evaluated_predicted_path_pub.publish(path_msg)

                    # Create a Path message for predictions
                    path_msg = Path()
                    path_msg.header.stamp = self.get_clock().now().to_msg()
                    path_msg.header.frame_id = msg.header.frame_id
                    # Fill the Path message with the actual future positions
                    for position in self.pose_buffer.evaluation_buffer:
                        pose_stamped = PoseStamped()
                        pose_stamped.header.stamp = self.get_clock().now().to_msg()
                        pose_stamped.header.frame_id = msg.header.frame_id
                        pose_stamped.pose.position.x = float(position[0])
                        pose_stamped.pose.position.y = float(position[1])
                        pose_stamped.pose.position.z = float(position[2])
                        # Assume no orientation information is available; quaternion set to identity
                        pose_stamped.pose.orientation.w = 1.0
                        path_msg.poses.append(pose_stamped)

                    # Publish the actual predicted Path message
                    self.actual_predicted_path_pub.publish(path_msg)

                    self.pose_buffer.reset_evaluation_trajectory()

                # Create a Path message
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = msg.header.frame_id

                # Add the initial position
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.header.frame_id = msg.header.frame_id
                pose_stamped.pose.position.x = float(position[0])
                pose_stamped.pose.position.y = float(position[1])
                pose_stamped.pose.position.z = float(position[2])
                # Assume no orientation information is available; quaternion set to identity
                pose_stamped.pose.orientation.w = 1.0
                path_msg.poses.append(pose_stamped)
                # Fill the Path message with the predicted positions
                for position in predictions:
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = self.get_clock().now().to_msg()
                    pose_stamped.header.frame_id = msg.header.frame_id
                    pose_stamped.pose.position.x = float(position[0])
                    pose_stamped.pose.position.y = float(position[1])
                    pose_stamped.pose.position.z = float(position[2])
                    # Assume no orientation information is available; quaternion set to identity
                    pose_stamped.pose.orientation.w = 1.0
                    path_msg.poses.append(pose_stamped)

                # Publish the predicted Path message
                self.path_publisher.publish(path_msg)
                
                path_msg = Path()
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.header.frame_id = msg.header.frame_id

                # Fill the Path message with the history positions
                for position in regularly_sampled_positions[-int(self.buffer_duration/self.dt):]:
                    pose_stamped = PoseStamped()
                    pose_stamped.header.stamp = self.get_clock().now().to_msg()
                    pose_stamped.header.frame_id = msg.header.frame_id
                    pose_stamped.pose.position.x = float(position[0])
                    pose_stamped.pose.position.y = float(position[1])
                    pose_stamped.pose.position.z = float(position[2])
                    # Assume no orientation information is available; quaternion set to identity
                    pose_stamped.pose.orientation.w = 1.0
                    path_msg.poses.append(pose_stamped)
                self.history_path_publisher.publish(path_msg)

                t1 = time.time()
                self.execution_timer_counter_ += 1
                self.aggregate_execution_time_ += (t1-t0)
                average_execution_time = self.aggregate_execution_time_ / self.execution_timer_counter_
                # self.get_logger().info(f'Average execution time: {average_execution_time} second(s)')
                # self.get_logger().info(f'Average execution frequency: {1/average_execution_time} Hz')


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
