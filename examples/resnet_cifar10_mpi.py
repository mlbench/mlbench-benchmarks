import argparse
import torch.distributed as dist


from mlbench_core.models.pytorch import Models
from mlbench_core.utils.pytorch.optim import Optimizer
from mlbench_core.utils.pytorch.lr import Scheduler
from mlbench_core.utils.pytorch.criterion import Criterion
from mlbench_core.utils.pytorch.metrics import Metrics
from mlbench_core.utils.pytorch.controlflow import ControlFlow
from mlbench_core.utils.pytorch.utils import initialize
from mlbench_core.dataset.imagerecognition.pytorch.dataloader import UniformPartitionedDataloader

# Fixed for closed division

dataset_config = {
    "dataset": "cifar10",
    "dataset_version": 1,
    "dataset_root": "/datasets",
    "num_parallel_workers": 2,
    "batch_size": 128,
    "shuffle_before_partition": True,
    "num_classes": 10
}

model_config = {
    "model": "resnet20",
    "model_version": 2,
}

utils_config = {
    "optim": "sgd",
    "nesterov": True,
    "weight_decay": 0,
    "momentum": 0.9,
    "lr": 0.1,
    "lr_scheduler": "const",
    "lr_scheduler_level": "epoch",
    "criterion": "CrossEntropyLoss",
    "metrics": "topk",

    "train_epochs": 164,
    "max_train_steps": 5,
    "max_batch_per_epoch": None,
    "resume": False,
    "runtime": {},

    "use_cuda": True,
    "dtype": "fp32",
    "transform_target_type": False,
    "validation": True,
    "seed": 42,
    "repartition_per_epoch": True
}

# configurations which does not influence accuracy.
meta_config = {
    "checkpoint": "all",
    "logging_level": "DEBUG",
    "logging_file": "/mlbench.log",
    "checkpoint_root": "/checkpoint",
    "comm_backend": "mpi",
    "cudnn_deterministic": False,
}


config = argparse.Namespace(**meta_config, **dataset_config, **model_config, **utils_config)

# TODO: Update it to a more general one
config.run_id = '1'
initialize(config)


def generate_dataloader(config):
    dataloader_train = UniformPartitionedDataloader.create(train=True, config=config)
    dataloader_val = UniformPartitionedDataloader.create(train=False, config=config)
    return dataloader_train, dataloader_val


# model = get_resnet_model(config)
model = Models.create(config)

# Create an optimizer associated with the model
optimizer = Optimizer.create(config, model)

# Create a learning rate scheduler for an optimizer
scheduler = Scheduler.create(config, optimizer)

# A criterion for computing the loss
criterion = Criterion.create(config, model)

# Metrics like Top 1/5 Accuracy
metrics = Metrics.create(config)

controlflow = ControlFlow.create(config)

controlflow(model=model, optimizer=optimizer, criterion=criterion,
            metrics=metrics, scheduler=scheduler, options=config,
            dataloader_fn=generate_dataloader)
