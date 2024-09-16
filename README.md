# drone_path_predictor_ros
This package is a ROS 2 implementaiton of our work entitled (Real-Time 3D UAV Trajectory Prediction Using Sequence-Based
Neural Models). This framework predicts accurate 3D drone trajectories in real time using Gated Recurrent Unit networks trained on synthetic dataset.

# Requirements
* Nvidia GPU. It also works on Nvidia Jetosn platform
* Tested with ROS humble
* Check the `setup.py` for more dependencies

# Execution

## Nodes
* There is currently a single node that can be launched using the `trajectory_predictor.launch.py` file.
* The node expects a pose array to get the position m easurements of the target drone. Currently, it consumes the first pose in the pose array. In the future, it might be used for multi-target trajectory prediciton. The pose array can be the output of a Kalman filter that provides state estimates of the target drone.
* The node will publish the predicted `Path` and `RMSE`/`MSE` for quantitaive analysis, as ROS topics.

## Parameters
All the parameters related to the GRU models can be found in the `config/trajectory_predictor.yaml` file.
