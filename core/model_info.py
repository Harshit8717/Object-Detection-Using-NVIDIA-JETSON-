import os
import torch


def get_parameter_count(model):

    total = 0
    trainable = 0

    for p in model.model.parameters():

        total += p.numel()

        if p.requires_grad:
            trainable += p.numel()

    return total, trainable


def human_readable_params(n):

    if n >= 1e9:
        return f"{n/1e9:.2f} Billion"

    if n >= 1e6:
        return f"{n/1e6:.2f} Million"

    if n >= 1e3:
        return f"{n/1e3:.2f} Thousand"

    return str(n)


def get_model_size(path):

    return round(os.path.getsize(path) / (1024**2), 2)


def get_model_info(model, model_path):

    info = {}

    total_params, trainable_params = get_parameter_count(model)

    info["Model Name"] = os.path.basename(model_path)

    info["Model Size (MB)"] = get_model_size(model_path)

    info["Total Parameters"] = total_params

    info["Readable Parameters"] = human_readable_params(total_params)

    info["Trainable Parameters"] = trainable_params

    info["Readable Trainable Parameters"] = human_readable_params(trainable_params)

    info["Layers"] = len(list(model.model.modules()))

    info["Classes"] = len(model.names)

    info["Class Names"] = model.names

    info["Device"] = str(next(model.model.parameters()).device)

    info["Precision"] = str(next(model.model.parameters()).dtype)

    info["Training Mode"] = model.model.training

    info["Input Shape"] = (1, 3, 640, 640)

    try:

        model.model.info()

    except:

        pass

    return info


def print_model_info(info):

    print()

    print("=" * 90)

    print("MODEL INFORMATION")

    print("=" * 90)

    for k, v in info.items():

        print(f"{k:<35}: {v}")
