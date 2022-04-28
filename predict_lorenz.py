import os
import argparse
import pdb
import time
import torch
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from model import load_model, save_model
from data.loaders import lorenz_loader


def encode_test(args, model):
    (train_loader, train_dataset, test_loader, test_dataset,) = lorenz_loader(args, num_workers=args.num_workers)
    import pdb; pdb.set_trace()
    z, _ = model.model.get_latent_representations(torch.from_numpy(test_dataset.get_full_data()[None,:,:]).to(args.device))
    return z


def main():
    parser = argparse.ArgumentParser(description='Lorenz experiment.')
    parser.add_argument('--out_dir', type=str, default="./result/lorenz")
    parser.add_argument('--experiment', type=str, default="lorenz")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--snr_index', type=int, default=0)
    # CPC
    parser.add_argument('--learning_rate', type=float, default=2.0e-4)
    parser.add_argument('--negative_samples', type=int, default=10)
    parser.add_argument('--prediction_step', type=int, default=12)
    parser.add_argument('--subsample', action="store_true")
    # General
    parser.add_argument('--genc_input', type=int, default=30)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--data_input_dir', type=str, default="./datasets/lorenz/lorenz_exploration.hdf5")
    parser.add_argument('--data_output_dir', type=str, default=".")
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--calc_accuracy', action="store_true")
    # Reload
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="result/lorenz")
    parser.add_argument('--model_num', type=int, default=10)

    args = parser.parse_args()

    # set start time
    args.time = time.ctime()

    # Device configuration
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args.current_epoch = args.start_epoch

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    model, optimizer = load_model(args, reload_model=True)

    z = encode_test(args, model.module)
    breakpoint()


if __name__ == "__main__":
    main()
