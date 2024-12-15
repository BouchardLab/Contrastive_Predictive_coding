import argparse
import time
import torch
import numpy as np
import json

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from model import load_model, save_model
from data.loaders import hc_loader
from utils.utils_prediction import linear_decode_r2


def encode_test(args, model):
    (train_loader, train_dataset, test_loader, test_dataset, Y_train, Y_test) = hc_loader(args,
                                                                                          num_workers=args.num_workers)
    X_train_cpc, _ = model.model.get_latent_representations(
        torch.from_numpy(train_dataset.get_full_data()[None, :, :]).float().to(args.device))
    X_train_cpc = X_train_cpc.to("cpu").detach().numpy()[0]
    X_test_cpc, _ = model.model.get_latent_representations(torch.from_numpy(test_dataset.get_full_data()[None, :, :]).float().to(args.device))
    X_test_cpc = X_test_cpc.to("cpu").detach().numpy()[0]
    # import pdb; pdb.set_trace()
    offsets = np.array([5, 10, 15])
    decoding_window = 3
    r2_cpcs = list()
    for offset in offsets:
        r2_cpc = linear_decode_r2(X_train_cpc, Y_train, X_test_cpc, Y_test, decoding_window=decoding_window,
                                  offset=offset)
        r2_cpcs.append(r2_cpc)
    return r2_cpcs


def main():
    parser = argparse.ArgumentParser(description='HC experiment.')
    parser.add_argument('--out_dir', type=str, default="./result/hc")
    parser.add_argument('--experiment', type=str, default="hc")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--snr_index', type=int, default=0)
    # CPC
    parser.add_argument('--learning_rate', type=float, default=2.0e-4)
    parser.add_argument('--negative_samples', type=int, default=10)
    parser.add_argument('--prediction_step', type=int, default=12)
    parser.add_argument('--subsample', action="store_true")
    parser.add_argument('--nonlinear_encoding', action="store_true")
    # General
    parser.add_argument('--genc_input', type=int, default=55)
    parser.add_argument('--seed', type=int, default=22)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--data_input_dir', type=str, default="/home/rui/Data/HC/example_data_hc.pickle")
    parser.add_argument('--data_output_dir', type=str, default=".")
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--calc_accuracy', action="store_true")
    # Reload
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--model_path', type=str, default="./result/hc")
    parser.add_argument('--model_num', type=int, default=0)

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

    r2_cpcs = encode_test(args, model.module)
    res = {"r2_cpcs": r2_cpcs}
    json.dump(res, open(args.out_dir + "/r2_cpcs.json", "w"))


if __name__ == "__main__":
    main()
