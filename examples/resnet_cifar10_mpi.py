r"""Example of using mlbench : CIFAR10 + Resnet20 + MPI"""

import types
import torch
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss

from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.models.pytorch.resnet import ResNetCIFAR
from mlbench_core.lr_scheduler.pytorch.lr import MultiStepLR
from mlbench_core.utils.pytorch.helpers import maybe_cuda
from mlbench_core.controlflow.pytorch import TrainValidation
from mlbench_core.dataset.imagerecognition.pytorch.dataloader import create_partition_transform_dataset


def generate_dataloader(train, config):
    r"""get train and validation dataloader."""
    dataset = create_partition_transform_dataset(train, config)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config.batch_size, shuffle=train,
        num_workers=config.num_parallel_workers,
        pin_memory=config.use_cuda, drop_last=False)
    return data_loader


def main(config):
    r"""Main logic."""
    model = maybe_cuda(ResNetCIFAR(20, False, 10, version=1), config)

    params_dict = dict(model.named_parameters())

    # Create an optimizer associated with the model
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                          weight_decay=config.weight_decay, nesterov=config.nesterov)

    # Create a learning rate scheduler for an optimizer
    scheduler = MultiStepLR(
        optimizer, milestones=config.multisteplr_milestones, gamma=config.multisteplr_gamma)

    # A loss_function for computing the loss
    loss_function = maybe_cuda(CrossEntropyLoss(), config)

    # Metrics like Top 1/5 Accuracy
    metrics = [TopKAccuracy(topk=int(m[3:])) for m in config.metrics]

    # controlflow = get_controlflow(config)
    # from debug_controlflow import TrainValidation
    controlflow = TrainValidation()

    controlflow(model=model, optimizer=optimizer, loss_function=loss_function,
                metrics=metrics, scheduler=scheduler, config=config,
                dataloader_fn=generate_dataloader)


if __name__ == '__main__':

    # Fixed for closed division

    DATASET_CONFIG = {
        "dataset": "cifar10",
        "dataset_version": 1,
        "dataset_root": "/datasets",
        "num_parallel_workers": 2,
        "batch_size": 128,
        "shuffle_before_partition": True,
        "num_classes": 10
    }

    MODEL_CONFIG = {
        "model": "resnet20",
        "model_version": 2,
        "average_models": True,  # Sum or average models between workers
    }

    UTILS_CONFIG = {
        "optim": "sgd",
        "nesterov": True,
        "weight_decay": 0.0001,
        "momentum": 0.9,
        "lr": 0.1,
        "lr_scheduler": "MultiStepLR",
        "multisteplr_milestones": [82, 109],
        "multisteplr_gamma": 0.1,
        "lr_scheduler_level": "epoch",
        "loss_function": "CrossEntropyLoss",
        "metrics": ["top1", "top5"],

        "train_epochs": 164,
        "max_train_steps": 164,
        "max_batch_per_epoch": None,
        "resume": False,
        "runtime": {},

        "use_cuda": True,
        "dtype": "fp32",
        "transform_target_type": False,
        "validation": True,
        "seed": 42,
        "repartition_per_epoch": True,
    }

    # configurations which does not influence accuracy.
    META_CONFIG = {
        "checkpoint": "all",
        "logging_level": "DEBUG",
        "logging_file": "/mlbench.log",
        "checkpoint_root": "/checkpoint",
        "comm_backend": "mpi",
        "cudnn_deterministic": False,
    }

    config = types.SimpleNamespace(
        **META_CONFIG, **DATASET_CONFIG, **MODEL_CONFIG, **UTILS_CONFIG)

    config.run_id = '1'
    initialize_backends(config)

    main(config)
