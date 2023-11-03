# trajectory_predictor.py
import numpy as np
import torch
import torch.nn as nn
import sys
import time

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device = {device}\n')

# The rest of your functions can also stay as is 
def load_normalization_parameters(npz_file_path):
    # Load the normalization parameters
    npz_file = np.load(npz_file_path)
    return npz_file['input_mean'], npz_file['input_std'], npz_file['target_mean'], npz_file['target_std']

def normalize_sequence(sequence, mean, std):
    return (sequence - mean) / std

def denormalize_sequence(sequence, mean, std):
    return (sequence * std) + mean

def predict_trajectory(model, sequence):
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_output = model(sequence_tensor)
        
    return predicted_output.squeeze(0).cpu().numpy()

def compute_velocity(position_sequence, dt):
    diff = np.diff(position_sequence, axis=0)
    velocity = diff / dt
    return velocity

def plot_on_ax(ax, inputs, predictions, actuals=None, title='', linestyle='-', color='blue'):
    ax.plot(inputs[:, 0], inputs[:, 1], inputs[:, 2], label='Input', linestyle=linestyle, color=color, linewidth=2)
    ax.plot(predictions[:, 0], predictions[:, 1], predictions[:, 2], label='Predicted', linestyle='--', color='red', linewidth=2)
    
    # Plot the actuals if provided
    if actuals is not None:
        ax.plot(actuals[:, 0], actuals[:, 1], actuals[:, 2], label='Actual', linestyle=':', color='green', linewidth=2)
    
    ax.set_title(title)
    
    min_x, max_x = min(inputs[:, 0].min(), predictions[:, 0].min(), (actuals[:, 0].min() if actuals is not None else predictions[:, 0].min())), max(inputs[:, 0].max(), predictions[:, 0].max(), (actuals[:, 0].max() if actuals is not None else predictions[:, 0].max()))
    min_y, max_y = min(inputs[:, 1].min(), predictions[:, 1].min(), (actuals[:, 1].min() if actuals is not None else predictions[:, 1].min())), max(inputs[:, 1].max(), predictions[:, 1].max(), (actuals[:, 1].max() if actuals is not None else predictions[:, 1].max()))
    min_z, max_z = min(inputs[:, 2].min(), predictions[:, 2].min(), (actuals[:, 2].min() if actuals is not None else predictions[:, 2].min())), max(inputs[:, 2].max(), predictions[:, 2].max(), (actuals[:, 2].max() if actuals is not None else predictions[:, 2].max()))
    
    ax.set_xlim(min_x - 5, max_x + 5)
    ax.set_ylim(min_y - 5, max_y + 5)
    ax.set_zlim(min_z - 5, max_z + 5)

def compute_predicted_positions(last_position, predicted_velocity, dt):
    predicted_positions = [last_position]
    for v in predicted_velocity:
        new_position = predicted_positions[-1] + v * dt
        predicted_positions.append(new_position)
    return np.array(predicted_positions)

# Trajectory Predictor Model Definition
class PositionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PositionPredictor, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), 10, hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out = self.fc(out)
        return out

# Velocity Predictor Model Definition
class VelocityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(VelocityPredictor, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), 10, hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out = self.fc(out)
        return out

class Predictor:
    def __init__(self, position_model_path,
                       velocity_model_path,
                       position_npz_path,
                       velocity_npz_path,
                       device, input_dim=3, hidden_dim=64, output_dim=3, num_layers=2):
        self.device = device
        
        # Load models and normalization parameters, similar to what you did in the if __name__ == '__main__': block
        self.position_model = PositionPredictor(input_dim, hidden_dim, output_dim, num_layers)
        self.position_model.load_state_dict(torch.load(position_model_path, map_location=self.device))
        self.position_model.to(self.device)
        self.position_model.eval()

        self.velocity_model = VelocityPredictor(input_dim, hidden_dim, output_dim, num_layers)
        self.velocity_model.load_state_dict(torch.load(velocity_model_path, map_location=self.device))
        self.velocity_model.to(self.device)
        self.velocity_model.eval()

        self.pos_input_mean, self.pos_input_std, self.pos_target_mean, self.pos_target_std = self.load_normalization_parameters(position_npz_path)
        self.vel_input_mean, self.vel_input_std, self.vel_target_mean, self.vel_target_std = self.load_normalization_parameters(velocity_npz_path)

    def predict_positions(self, sequence, dt):
        # Normalize the sequence using the loaded mean and std
        normalized_position_sequence = normalize_sequence(sequence[:-1], self.pos_input_mean, self.pos_input_std)
        # Perform prediction using the loaded models
        predicted_normalized_position = predict_trajectory(self.position_model, normalized_position_sequence)
        # Denormalize the predictions
        predicted_positions = denormalize_sequence(predicted_normalized_position, self.pos_target_mean, self.pos_target_std)
        
        return predicted_positions
    
    def predict_positions_from_velocity(self, pos_sequence, dt):
        # Compute velocity from the 21-point input position sequence
        input_velocity = compute_velocity(pos_sequence, dt=0.1)
        # Normalize the input sequence for the velocity model
        normalized_velocity_input = normalize_sequence(input_velocity, self.vel_input_mean, self.vel_input_std)
        predicted_normalized_velocity_output = predict_trajectory(self.velocity_model, normalized_velocity_input)
        # Denormalize the predicted velocity trajectory
        predicted_velocity_output = denormalize_sequence(predicted_normalized_velocity_output, self.vel_target_mean, self.vel_target_std)
        # Compute the predicted position using the last point in the input sequence, predicted velocities and dt
        predicted_positions_from_velocity = compute_predicted_positions(pos_sequence[-1], predicted_velocity_output, dt)

        return predicted_positions_from_velocity

