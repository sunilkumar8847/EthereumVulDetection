import argparse


def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='Smart Contracts Vulnerability Detection')

    parser.add_argument('-D', '--dataset', type=str, default='', choices=[])
    parser.add_argument('-M', '--model', type=str, default='EncoderWeight',
                        choices=['EncoderWeight', 'EncoderAttention', 'FNNModel'])
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('-v', '--vulnerability_types', type=str, default='reentrancy',
                        help='Comma-separated list of vulnerability types to detect (reentrancy,timestamp)')
    parser.add_argument('--save_model', action='store_true', help='Save the trained model')

    return parser.parse_args()
