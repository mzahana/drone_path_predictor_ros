from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('drone_path_predictor_ros'), 'config')
    default_param_file = os.path.join(config_dir, 'trajectory_predictor.yaml')
    
    # Declare the arguments for remapping
    pose_topic_argument = DeclareLaunchArgument(
        'pose_topic',
        default_value='/kf/good_tracks_pose_array',
        description='The input topic with PoseArray messages.'
    )
    path_topic_argument = DeclareLaunchArgument(
        'path_topic',
        default_value='out/gru_predicted_path',
        description='The output topic for Path messages.'
    )
    param_file_argument = DeclareLaunchArgument(
        'param_file',
        default_value=default_param_file,
        description='Path to the parametetrs file.'
    )
    gru_namespace_argument = DeclareLaunchArgument(
        'gru_namespace',
        default_value='',
        description='Node namespace.'
    )
    
    return LaunchDescription([
        pose_topic_argument,
        path_topic_argument,
        param_file_argument,
        gru_namespace_argument,
        
        Node(
            package='drone_path_predictor_ros',
            executable='trajectory_predictor_node',
            name='trajectory_predictor_node',
            namespace=LaunchConfiguration('gru_namespace'),
            output='screen',
            parameters=[LaunchConfiguration('param_file')],
            remappings=[
                ('in/pose_array', LaunchConfiguration('pose_topic')),
                ('out/gru_predicted_path', LaunchConfiguration('path_topic')),
                ('out/gru_history_path','out/gru_history_path'),
                
            ]
        )
    ])
