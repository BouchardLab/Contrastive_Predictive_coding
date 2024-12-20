import os
import argparse
import time
import torch
import numpy as np

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
from model import load_model, save_model
from data.loaders import lorenz_loader


def train(args, model, optimizer, writer):

    # import pdb; pdb.set_trace()
    # get datasets and dataloaders
    (train_loader, train_dataset, test_loader, test_dataset, X_dynamics) = lorenz_loader(args, num_workers=args.num_workers)

    total_step = len(train_loader)
    print_idx = 100

    # at which step to validate training
    # validation_idx = 1000

    best_loss = 0

    start_time = time.time()
    global_step = 0
    for epoch in range(args.start_epoch, args.start_epoch + args.num_epochs):
        loss_epoch = 0

        for step, lorenz in enumerate(train_loader):
            # import pdb; pdb.set_trace()
            start_time = time.time()

            # if step % validation_idx == 0:
            #     validate_speakers(args, train_dataset, model, optimizer, epoch, step, global_step, writer)

            lorenz = lorenz.to(args.device)

            # forward
            loss = model(lorenz)

            # accumulate losses for all GPUs
            loss = loss.mean()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # backward, depending on mixed-precision
            model.zero_grad()
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            optimizer.step()

            if step % print_idx == 0:
                examples_per_second = args.batch_size / (time.time() - start_time)
                print(
                    "[Epoch {}/{}] Train step {:04d}/{:04d} \t Examples/s = {:.2f} \t "
                    "Loss = {:.4f} \t Time/step = {:.4f}".format(
                        epoch,
                        args.num_epochs,
                        step,
                        len(train_loader),
                        examples_per_second,
                        loss,
                        time.time() - start_time,
                    )
                )

            writer.add_scalar("Loss/train_step", loss, global_step)
            loss_epoch += loss
            global_step += 1

        avg_loss = loss_epoch / len(train_loader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        # ex.log_scalar("loss.train", avg_loss, epoch)

        conv = 0
        for idx, layer in enumerate(model.module.model.modules()):
            if isinstance(layer, torch.nn.Conv1d):
                writer.add_histogram(
                    "Conv/weights-{}".format(conv),
                    layer.weight,
                    global_step=global_step,
                )
                conv += 1

            if isinstance(layer, torch.nn.GRU):
                writer.add_histogram(
                    "GRU/weight_ih_l0", layer.weight_ih_l0, global_step=global_step
                )
                writer.add_histogram(
                    "GRU/weight_hh_l0", layer.weight_hh_l0, global_step=global_step
                )

        if avg_loss > best_loss:
            best_loss = avg_loss
            save_model(args, model, optimizer, best=True)

        # save current model state
        save_model(args, model, optimizer)
        args.current_epoch += 1


def main():
    parser = argparse.ArgumentParser(description='Lorenz experiment.')
    parser.add_argument('--out_dir', type=str, default="./result")
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
    parser.add_argument('--model_path', type=str, default=".")
    parser.add_argument('--model_num', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # set start time
    args.time = time.ctime()

    # Device configuration
    if args.device is None:
        args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    args.current_epoch = args.start_epoch

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load model
    model, optimizer = load_model(args)

    # initialize TensorBoard
    tb_dir = os.path.join(args.out_dir, args.experiment, str(args.snr_index))
    if not os.path.exists(tb_dir):
        os.makedirs(tb_dir)
    writer = SummaryWriter(log_dir=tb_dir)
    # writer.add_graph(model.module, torch.rand(args.batch_size, 1, 20480).to(args.device))

    try:
        train(args, model, optimizer, writer)
    except KeyboardInterrupt:
        print("Interrupting training, saving model")

    save_model(args, model, optimizer)


if __name__ == "__main__":
    main()
