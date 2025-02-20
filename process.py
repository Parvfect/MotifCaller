# Recreate as the model config file
import argparse
import os 
from datetime import datetime
from training_loop import main


parser = argparse.ArgumentParser(
                    prog='Motif Caller',
                    description='Training the motif caller',
                    epilog='Use parser to set some parameters for training')

parser.add_argument('--epochs', type=int, help="Training epochs")
parser.add_argument('--sampling_rate', type=float, help="Sample the data")
parser.add_argument('--window_size', type=int, help="Window size for ctc")
parser.add_argument('--window_step', type=int, help="Step size for window")
parser.add_argument('--running_on_hpc', action='store_true', help="Flag for running on hpc")

parser.set_defaults(
    epochs=50, window_size=1024, window_step=800, sampling_rate=1.0, running_on_hpc=False)

args = parser.parse_args()

if __name__ == '__main__':
    epochs = args.epochs
    sampling_rate = args.sampling_rate
    window_size = args.window_size
    window_step = args.window_step
    running_on_hpc = args.running_on_hpc

    main(epochs=epochs, sampling_rate=sampling_rate, window_size=window_size,
        window_step=window_step, running_on_hpc=running_on_hpc)
