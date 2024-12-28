# drone_path_predictor_ros

This is a ROS 2 package for real-time UAV trajectory prediction using a Gated Recurrent Unit (GRU)-based neural network. It implements our approach described in the following publication:

> **VECTOR: Velocity-Enhanced GRU Neural Network for Real-Time 3D UAV Trajectory Prediction**  
> Omer Nacar, Mohamed Abdelkader, Lahouari Ghouti, Kahled Gabr, Abdulrahman Al-Batati, Anis Koubaa, *Drones*, 9(1), 8, 2025.  
> DOI: [10.3390/drones9010008](https://doi.org/10.3390/drones9010008)

In this work, we propose a trajectory prediction pipeline for UAVs using GRU-based neural networks. Our approach allows for enhanced prediction accuracy and robustness by leveraging both position and velocity estimates, improving upon traditional approaches that rely solely on position.

**If you find this repo useful, kindly give it a STAR :)**

---
## Citing This Work

If you use this package in your work, please cite the corresponding publication:

```bibtex
@Article{drones9010008,
  AUTHOR = {Nacar, Omer and Abdelkader, Mohamed and Ghouti, Lahouari and Gabr, Kahled and Al-Batati, Abdulrahman and Koubaa, Anis},
  TITLE = {VECTOR: Velocity-Enhanced GRU Neural Network for Real-Time 3D UAV Trajectory Prediction},
  JOURNAL = {Drones},
  VOLUME = {9},
  YEAR = {2025},
  NUMBER = {1},
  ARTICLE-NUMBER = {8},
  URL = {https://www.mdpi.com/2504-446X/9/1/8},
  ISSN = {2504-446X},
  DOI = {10.3390/drones9010008}
}
```

---

## Features

- **Real-Time 3D UAV Trajectory Prediction**  
  Predicts future UAV trajectories based on incoming pose (and optionally velocity) data.
- **GRU-Based Model**  
  Uses a Gated Recurrent Unit (GRU) network to capture long-term dependencies and dynamics.
- **Configurable Parameters**  
  Load your own trained PyTorch models, statistics, and tune hyperparameters via ROS 2 parameters.
- **ROS 2 Integration**  
  Subscribes to `PoseArray` messages and publishes predicted `Path` messages, which can be visualized with standard ROS 2 tools (e.g., RViz).

---

## Contents

- **`drone_path_predictor_ros/trajectory_predictor_node.py`**  
  ROS 2 node that buffers incoming poses, runs the GRU-based prediction, and publishes the predicted path.
- **`drone_path_predictor_ros/pose_buffer.py`**  
  Handles buffering of incoming pose data and sampling them at a regular rate.
- **`drone_path_predictor_ros/trajectory_predictor.py`**  
  Loads and runs the trained GRU models (position and optional velocity models).
- **`launch/trajectory_predictor.launch.py`**  
  Example launch file to start the node with parameters loaded from a YAML file.
- **`config/trajectory_predictor.yaml`**  
  YAML file containing all the configurable parameters for the predictor node, including paths to the models and stats.

---

## Installation

1. **Clone this repository into your ROS 2 workspace** (e.g., `~/ros2_ws/src`):
   ```bash
   cd ~/ros2_ws/src
   git clone https://github.com/mzahana/drone_path_predictor_ros.git
   ```
   
2. **Install required Python dependencies**:
   ```bash
   # In case you have custom dependencies for PyTorch, TorchVision, etc.
   # Example:
   pip3 install torch torchvision  # or as needed for your environment
   ```

3. **Build your workspace**:
   ```bash
   cd ~/ros2_ws
   colcon build
   ```
   
4. **Source the workspace**:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```
   
---

## Usage

### Launching the Predictor

Use the provided launch file to run the predictor node:

```bash
ros2 launch drone_path_predictor_ros trajectory_predictor.launch.py
```

---

## Node Description

### `trajectory_predictor_node`

This node:

1. Subscribes to a `PoseArray` topic (by default `in/pose_array`).
2. Buffers incoming pose data for a specified duration (`buffer_duration`) and samples it at a given rate (`dt`).
3. Uses the loaded GRU models (position and, optionally, velocity) to predict a future trajectory.
4. Publishes:
   - The **predicted path** (`out/gru_predicted_path`).
   - The **historical path** from the buffer (`out/gru_history_path`).
   - **Actual future path** (`out/actual_predicted_path`) for evaluation.
   - **Evaluated predicted path** (`out/evaluated_predicted_path`) along with MSE and RMSE metrics.

---

## Topics

Below is a description of the subscribed and published topics in a tabular format:

| **Topic Name**                     | **Type**                       | **Direction** | **Description**                                                                                                  |
|------------------------------------|--------------------------------|---------------|------------------------------------------------------------------------------------------------------------------|
| `in/pose_array`                    | `geometry_msgs/PoseArray`      | Subscribed    | Receives pose measurements, typically from a localization or object tracking pipeline.                           |
| `out/gru_predicted_path`          | `nav_msgs/Path`                | Published     | Predicted UAV trajectory.                                                                                       |
| `out/gru_history_path`            | `nav_msgs/Path`                | Published     | History of the last buffered poses.                                                                             |
| `out/actual_predicted_path`       | `nav_msgs/Path`                | Published     | Ground truth path of the next poses (used for evaluation).                                                      |
| `out/evaluated_predicted_path`    | `nav_msgs/Path`                | Published     | The predictions being evaluated (aligned with `actual_predicted_path`).                                         |
| `out/evaluation_mse`              | `std_msgs/Float32`             | Published     | MSE metric for the predictions.                                                                                 |
| `out/evaluation_rmse`             | `std_msgs/Float32`             | Published     | RMSE metric for the predictions.                                                                                |

---

## Parameters

All parameters can be configured in the [trajectory_predictor.yaml](config/trajectory_predictor.yaml) file or overridden in the launch file. Below is a tabular overview:

| **Parameter**             | **Type**  | **Default**                   | **Description**                                                                                                                       |
|---------------------------|-----------|--------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `position_model_path`     | `string`  | `"path/to/your/position_model.pt"` | Path to the position GRU model.                                                                                                     |
| `velocity_model_path`     | `string`  | `"path/to/your/velocity_model.pt"` | Path to the velocity GRU model.                                                                                                     |
| `position_stats_file`     | `string`  | `"position_stats_file"`           | Path to the JSON file containing mean/std for position data.                                                                       |
| `velocity_stats_file`     | `string`  | `"velocity_stats_file"`           | Path to the JSON file containing mean/std for velocity data.                                                                       |
| `buffer_duration`         | `double`  | `2.0`                            | Duration (in seconds) of the pose buffer.                                                                                           |
| `dt`                      | `double`  | `0.1`                            | Time step for sampling future predictions.                                                                                          |
| `pos_hidden_dim`          | `int`     | `64`                             | Hidden dimension for the position GRU model.                                                                                        |
| `pos_num_layers`          | `int`     | `2`                              | Number of layers for the position GRU model.                                                                                        |
| `pos_dropout`             | `double`  | `0.5`                            | Dropout rate for the position GRU model.                                                                                            |
| `vel_hidden_dim`          | `int`     | `64`                             | Hidden dimension for the velocity GRU model.                                                                                        |
| `vel_num_layers`          | `int`     | `2`                              | Number of layers for the velocity GRU model.                                                                                        |
| `vel_dropout`             | `double`  | `0.5`                            | Dropout rate for the velocity GRU model.                                                                                            |
| `use_velocity_prediction` | `bool`    | `False`                          | Whether to enable velocity-based prediction.                                                                                        |
| `use_whitening`           | `bool`    | `False`                          | Whether to enable data whitening (normalization) based on the provided statistics.                                                  |

---

## Visualization

To visualize the predicted path in RViz2:

1. Launch RViz2:
   ```bash
   ros2 run rviz2 rviz2
   ```
2. Add **Path** displays for:
   - `out/gru_predicted_path`
   - `out/gru_history_path`
   - `out/actual_predicted_path`
   - `out/evaluated_predicted_path`
   
You will see the UAV’s historical path in one color, the predicted future path in another, and (if you have ground truth or concurrent measurement data) the actual future path for evaluation.

---


## Contact

For any inquiries or issues, feel free to open an issue on this repository.
