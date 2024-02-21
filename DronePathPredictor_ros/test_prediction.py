import argparse
import numpy as np
import torch
from trajectory_predictor import load_normalization_parameters, normalize_sequence, denormalize_sequence, Predictor
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test GRU model for trajectory prediction.')
    parser.add_argument('--model_path_pos', type=str, required=True, help='Path to the position GRU model file (.pth)')
    parser.add_argument('--model_path_vel', type=str, required=True, help='Path to the velocity GRU model file (.pth)')
    parser.add_argument('--pos_stats_file', type=str, required=True, help='Path to the position stats (.npz) file')
    parser.add_argument('--vel_stats_file', type=str, required=True, help='Path to the velocity stats (.npz) file')
    parser.add_argument('--dataset_pos', type=str, required=True, help='Path to the position dataset file (.npz)')
    parser.add_argument('--dataset_vel', type=str, required=True, help='Path to the velocity dataset file (.npz)')
    parser.add_argument('--test_type', type=str, required=True, choices=['position', 'velocity', 'both'], help='Test type: position, velocity, or both')
    return parser.parse_args()

def load_data(dataset_path):
    # Load dataset (modify based on your dataset format)
    data = np.load(dataset_path)
    input_segments = data['input_segments']  # assuming the input segments are stored under the key 'inputs'
    output_segments = data['output_segments']  # assuming the output segments are stored under the key 'outputs'
    return input_segments, output_segments

def select_random_sample(input_segments, output_segments):
    index = random.randint(0, input_segments.shape[2] - 1)
    return input_segments[:, :, index], output_segments[:, :, index]

def plot_sequences(input_seq, actual_seq, predicted_seq, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each sequence
    ax.plot(input_seq[:, 0], input_seq[:, 1], input_seq[:, 2], label='Input')
    ax.plot(actual_seq[:, 0], actual_seq[:, 1], actual_seq[:, 2], label='Actual')
    ax.plot(predicted_seq[:, 0], predicted_seq[:, 1], predicted_seq[:, 2], label='Predicted')

    # Set labels for axes
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    # Set title and legend
    plt.title(title)
    ax.legend()

    plt.show()

def main():
    args = parse_arguments()

    # Load datasets and select random samples
    if args.test_type in ['position', 'both']:
        pos_inp_segments, pos_out_segments = load_data(args.dataset_pos)
        sample_pos_inp, sample_pos_out = select_random_sample(pos_inp_segments, pos_out_segments)
        print('Loaded position dataset')
        print(f'Shape of sample_pos_inp: {sample_pos_inp.shape}')
        print(f'Shape of sample_pos_out: {sample_pos_out.shape}')

    if args.test_type in ['velocity', 'both']:
        vel_inp_segments, vel_out_segments = load_data(args.dataset_vel)
        sample_vel_inp, sample_vel_out = select_random_sample(vel_inp_segments, vel_out_segments)
        print('Loaded velocity dataset')
        print(f'Shape of sample_vel_inp: {sample_vel_inp.shape}')
        print(f'Shape of sample_vel_out: {sample_vel_out.shape}')


    # Models
    models = Predictor(args.model_path_pos, args.model_path_vel, args.pos_stats_file, args.vel_stats_file,
                       pos_hidden_dim=256, pos_num_layers=2,  pos_dropout=0.5,
                         vel_hidden_dim=256, vel_num_layers=2,  vel_dropout=0.5)

    print('Created models object')

    # Prediction and plotting for position
    if args.test_type in ['position', 'both']:
        predicted_pos = models.predict_positions(sample_pos_inp.T)
        print('position is predicted')
        plot_sequences(sample_pos_inp.T, sample_pos_out.T, predicted_pos, 'Position')

        sample_pos_inp = np.hstack((sample_pos_inp, sample_pos_inp[:, -1].reshape(3, 1)))
        pos_from_vel = models.predict_positions_from_velocity(sample_pos_inp.T, 0.1)
        plot_sequences(sample_pos_inp.T, sample_pos_out.T, pos_from_vel, 'Position from velocity')

    # Prediction and plotting for velocity
    if args.test_type in ['velocity', 'both']:
        predicted_vel = models.predict_velocity(sample_vel_inp.T)
        print('velocity is predicted')
        plot_sequences(sample_vel_inp.T, sample_vel_out.T, predicted_vel, 'Velocity')

if __name__ == '__main__':
    main()
