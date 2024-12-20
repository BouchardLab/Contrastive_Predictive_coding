import os
import torch
from modules import AudioModel, GeneralModel

try:
    from apex import amp
except ImportError:
    raise ImportError(
        "Install the apex package from https://www.github.com/nvidia/apex to use fp16 for training"
    )


def audio_model(args):
    strides = [5, 4, 2, 2, 2]
    filter_sizes = [10, 8, 4, 4, 4]
    padding = [2, 2, 2, 2, 1]
    genc_hidden = 512
    gar_hidden = 256

    model = AudioModel(
        args,
        strides=strides,
        filter_sizes=filter_sizes,
        padding=padding,
        genc_hidden=genc_hidden,
        gar_hidden=gar_hidden,
    )
    return model


def lorenz_model(args):
    genc_hidden = 3
    gar_hidden = 3
    model = GeneralModel(args, genc_hidden=genc_hidden, gar_hidden=gar_hidden)
    return model

def general_model(args):
    genc_hidden = 5
    gar_hidden = 5
    model = GeneralModel(args, genc_hidden=genc_hidden, gar_hidden=gar_hidden)
    return model

m1_model = hc_model = temp_model = ms_model = mc_model = general_model    

def load_model(args, reload_model=False):

    if args.experiment == "audio":
        model = audio_model(args)
    elif args.experiment == "lorenz":
        model = lorenz_model(args)
    elif args.experiment == "m1":
        model = m1_model(args)
    elif args.experiment == "hc":
        model = hc_model(args)
    elif args.experiment == "temp":
        model = temp_model(args)
    elif args.experiment == "ms":
        model = ms_model(args)
    elif args.experiment == "mc":
        model = mc_model(args)
    else:
        raise NotImplementedError

    # reload model
    if args.start_epoch > 0 or reload_model:
        if args.start_epoch == 0:
            load_epoch = args.model_num
        else:
            load_epoch = args.start_epoch

        print("### RELOADING MODEL FROM CHECKPOINT {} ###".format(load_epoch))
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(load_epoch))
        model.load_state_dict(torch.load(model_fp))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.fp16:
        print("### USING FP16 ###")
        model, optimizer = amp.initialize(model.to(args.device), optimizer, opt_level=args.fp16_opt_level)
    else:
        model = model.to(args.device)
        # model, optimizer = amp.initialize(model.to(args.device), optimizer, opt_level="O0")

    args.num_gpu = torch.cuda.device_count()
    print("Using {} GPUs".format(args.num_gpu))

    model = torch.nn.DataParallel(model)
    args.batch_size = args.batch_size * args.num_gpu

    # import pdb; pdb.set_trace()

    return model, optimizer


def save_model(args, model, optimizer, best=False):

    if best:
        out = os.path.join(args.out_dir, "best_checkpoint.tar")
    else:
        out = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    torch.save(model.module.state_dict(), out)

    with open(os.path.join(args.out_dir, "best_checkpoint.txt"), "w") as f:
        f.write(str(args.current_epoch))

