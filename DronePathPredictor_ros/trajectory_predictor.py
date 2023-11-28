#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import sys
import time

# Check if GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f'device = {device}\n')

# The rest of your functions can also stay as is 
def load_normalization_parameters(npz_file_path):
    # Load the normalization parameters
    npz_file = np.load(npz_file_path)
    return npz_file['input_mean'], npz_file['input_std'], npz_file['target_mean'], npz_file['target_std']

def load_pos_stats(filename):
    with np.load(filename, allow_pickle=True) as data:
        mean = data['input_mean']
        L_matrix = data['L_matrix']
        max_length = data['max_length']
    return mean, L_matrix, max_length

def load_vel_stats(filename):
    with np.load(filename, allow_pickle=True) as data:
        mean = data['input_mean']
        L_matrix = data['L_matrix']
        max_velocity = data['max_velocity']
    return mean, L_matrix, max_velocity

def load_mean_and_L(filename):
    """
    Loads the mean and L matrix from a given npz file.

    :param filename: Path to the .npz file containing the stats.
    :return: Tuple containing the mean and L matrix.
    """
    with np.load(filename, allow_pickle=True) as data:
        mean = data['input_mean']
        L_matrix = data['L_matrix']
    return mean, L_matrix

def normalize_sequence(sequence, mean, std):
    return (sequence - mean) / std

def denormalize_sequence(sequence, mean, std):
    return (sequence * std) + mean

def max_normalize_sequence(sequence, max):
    return sequence / max

def max_denormalize_sequence(sequence, max):
    return sequence * max

def whiten_3d_sequence(sequence, mean, L):
    # Assuming sequence of shape (N,3)
    # Apply whitening transformation
    whitened_sequence = np.dot(sequence - mean, L.T)
    return whitened_sequence

def dewhiten_3d_sequence(whitened_sequence, mean, L):
    # Assuming sequence of shape (N,3)
    # Apply dewhitening transformation
    L_inv = np.linalg.inv(L)
    dewhitened_sequence = np.dot(whitened_sequence, L_inv.T) + mean
    return dewhitened_sequence


def predict_trajectory(model, sequence, device):
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_output = model(sequence_tensor)
        
    return predicted_output.squeeze(0).cpu().numpy()

def compute_velocity(position_sequence, dt):
    diff = np.diff(position_sequence, axis=0)
    velocity = diff / dt
    return velocity

def plot_on_ax(self, ax, inputs, predictions, actuals=None, title='', linestyle='-', color='blue'):
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
        # print("v: \n", v)
        # print("last position: \n", predicted_positions[-1])
        new_position = predicted_positions[-1] + (v * dt)
        # print("new_position: \n", new_position)
        predicted_positions.append(new_position)
    return np.array(predicted_positions)

# Trajectory Predictor Model Definition
class PositionPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PositionPredictor, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), 10, self.hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out = self.fc(out)
        return out

# Trajectory Predictor Model Definition
class PositionPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(PositionPredictor2, self).__init__()

        # Initialize the GRUs
        self.grus = nn.ModuleList([nn.GRU(input_dim if i == 0 else hidden_dim, 
                                          hidden_dim, 
                                          num_layers, 
                                          batch_first=True) 
                                   for i in range(num_layers)])

        # The final fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Store the hidden dimension to use later
        self.hidden_dim = hidden_dim

    def forward(self, x):
        h_n = None
        # Pass through each of the GRU layers
        for i, gru in enumerate(self.grus):
            out, h_n = gru(x if i == 0 else dec_input, h_n)
            if i < len(self.grus) - 1:  # Prepare dec_input if not last GRU layer
                dec_input = torch.zeros(x.size(0), out.size(1), self.hidden_dim).to(x.device)
        
        # Pass the output of the last GRU layer through the fully connected layer
        out = self.fc(out)
        return out
    
class PositionPredictor3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(PositionPredictor3, self).__init__()
        
        self.hidden_dim = hidden_dim  # Add this line to set hidden_dim as an instance attribute

        # Encoding layer with dropout
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Decoding layer
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Encoding
        out, h_n = self.gru1(x)

        # Decoding
        # We'll initialize the decoder's first input as zeros.
        dec_input = torch.zeros(x.size(0), 10, self.hidden_dim).to(x.device)  # Assuming batch_first is True
        out, _ = self.gru2(dec_input, h_n)

        # Predict the next 10 points
        out = self.fc(out)

        return out
    
# Velocity Predictor Model Definition
class VelocityPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(VelocityPredictor, self).__init__()

        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        out, h_n = self.gru1(x)
        dec_input = torch.zeros(x.size(0), 10, self.hidden_dim).to(x.device)
        out, _ = self.gru2(dec_input, h_n)
        out = self.fc(out)
        return out

class VelocityPredictor2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.5):
        super(VelocityPredictor2, self).__init__()
        
        self.hidden_dim = hidden_dim  # Add this line to set hidden_dim as an instance attribute

        # Encoding layer with dropout
        self.gru1 = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Decoding layer
        self.gru2 = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Encoding
        out, h_n = self.gru1(x)

        # Decoding
        # We'll initialize the decoder's first input as zeros.
        dec_input = torch.zeros(x.size(0), 10, self.hidden_dim).to(x.device)  # Assuming batch_first is True
        out, _ = self.gru2(dec_input, h_n)

        # Predict the next 10 points
        out = self.fc(out)

        return out

class Predictor:
    def __init__(self, position_model_path,
                       velocity_model_path,
                       position_stats_file,
                       velocity_stats_file,
                       pos_input_dim=3, pos_hidden_dim=64, pos_output_dim=3, pos_num_layers=2, pos_input_length=21, pos_output_length=10, pos_dropout=0.5,
                       vel_input_dim=3, vel_hidden_dim=64, vel_output_dim=3, vel_num_layers=2, vel_input_length=21, vel_output_length=10, vel_dropout=0.5,
                       use_whitening=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and normalization parameters, similar to what you did in the if __name__ == '__main__': block
        # checkpoint = torch.load(position_model_path)
        # Extract the state dictionary
        # model_state_dict = checkpoint['state_dict']
        # self.position_model = PositionPredictor(pos_input_dim, pos_hidden_dim, pos_output_dim, pos_num_layers)
        self.position_model = PositionPredictor3(pos_input_dim, pos_hidden_dim, pos_output_dim, pos_num_layers, pos_dropout)
        self.position_model.load_state_dict(torch.load(position_model_path, map_location=self.device))
        self.position_model.to(self.device)
        # self.position_model.load_state_dict(model_state_dict)
        self.position_model.eval()

        # checkpoint = torch.load(velocity_model_path)
        # Extract the state dictionary
        # model_state_dict = checkpoint['state_dict']
        # self.velocity_model = VelocityPredictor(vel_input_dim, vel_hidden_dim, vel_output_dim, vel_num_layers)
        self.velocity_model = VelocityPredictor2(vel_input_dim, vel_hidden_dim, vel_output_dim, vel_num_layers, vel_dropout)
        self.velocity_model.load_state_dict(torch.load(velocity_model_path, map_location=self.device))
        self.velocity_model.to(self.device)
        # self.velocity_model.load_state_dict(model_state_dict)
        self.velocity_model.eval()

        self.pos_mean, self.pos_L_matrix, self.pos_max = load_pos_stats(position_stats_file)
        self.vel_mean, self.vel_L_matrix, self.vel_max = load_vel_stats(velocity_stats_file)
        print(f"pos_mean: {self.pos_mean}\npos_L_matrix: {self.pos_L_matrix}\npos_max: {self.pos_max}")
        print(f"vel_mean: {self.vel_mean}\nvel_L_matrix: {self.vel_L_matrix}\nvel_max: {self.vel_max}")

        self.use_whitening=use_whitening

    def predict_positions(self, sequence):
        # Perform prediction using the loaded models
        if self.use_whitening:
            # Normalize the sequence using the loaded mean and and L matrix
            whitened_pos_seq = whiten_3d_sequence(sequence, self.pos_mean, self.pos_L_matrix)
            predicted_normalized_position = predict_trajectory(self.position_model, whitened_pos_seq, self.device)
            # Denormalize the predictions
            predicted_positions = dewhiten_3d_sequence(predicted_normalized_position, self.pos_mean, self.pos_L_matrix)
        else:
            # Normalize the sequence using the max position length
            normalized_position_sequence = max_normalize_sequence(sequence, self.pos_max)
            predicted_normalized_position = predict_trajectory(self.position_model, normalized_position_sequence, self.device)
            # Denormalize the predictions
            predicted_positions = max_denormalize_sequence(predicted_normalized_position, self.pos_max)
        
        return predicted_positions.copy()
    
    def predict_velocity(self, sequence):
        if self.use_whitening:
            # Normalize the sequence using the loaded mean and and L matrix
            whitened_vel_seq = whiten_3d_sequence(sequence, self.vel_mean, self.vel_L_matrix)
            predicted_normalized_velocity = predict_trajectory(self.velocity_model, whitened_vel_seq, self.device)
            # Denormalize the predictions
            predicted_velocity = dewhiten_3d_sequence(predicted_normalized_velocity, self.vel_mean, self.vel_L_matrix)
        else:
            # Normalize the sequence using the loaded mean and std
            normalized_velocity_sequence = max_normalize_sequence(sequence, self.vel_max)
            predicted_normalized_velocity = predict_trajectory(self.velocity_model, normalized_velocity_sequence, self.device)
            # Denormalize the predictions
            predicted_velocity = max_denormalize_sequence(predicted_normalized_velocity,self.vel_max)
        
        return predicted_velocity.copy()
    
    def predict_positions_from_velocity(self, pos_sequence, dt):
        # Compute velocity from the input position sequence
        input_velocity = compute_velocity(pos_sequence, dt)
         
        # Getting the last row
        last_vel = input_velocity[-1]

        # Appending the last row to the array
        extedned_vel = np.append(input_velocity, [last_vel], axis=0)

        predicted_velocity_output = self.predict_velocity(extedned_vel)

        # Compute the predicted position using the last point in the input sequence, predicted velocities and dt
        predicted_positions_from_velocity = compute_predicted_positions(pos_sequence[-1], predicted_velocity_output, dt)

        return predicted_positions_from_velocity.copy()

