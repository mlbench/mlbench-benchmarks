import types


from mlbench_core.models.pytorch import get_model
from mlbench_core.optim.pytorch import get_optimizer
from mlbench_core.lr_scheduler.pytorch import get_scheduler
from mlbench_core.evaluation.pytorch import get_loss_function
from mlbench_core.evaluation.pytorch import get_metrics
from mlbench_core.controlflow.pytorch import get_controlflow
from mlbench_core.utils.pytorch import initialize_backends

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
    "average_models": True,  # Sum or average models between workers
}

utils_config = {
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
meta_config = {
    "checkpoint": "all",
    "logging_level": "DEBUG",
    "logging_file": "/mlbench.log",
    "checkpoint_root": "/checkpoint",
    "comm_backend": "mpi",
    "cudnn_deterministic": False,
}


config = types.SimpleNamespace(**meta_config, **dataset_config, **model_config, **utils_config)

config.run_id = '1'
initialize_backends(config)


def generate_dataloader(config):
    dataloader_train = UniformPartitionedDataloader.create(train=True, config=config)
    dataloader_val = UniformPartitionedDataloader.create(train=False, config=config)
    return dataloader_train, dataloader_val


model = get_model(config)

# Create an optimizer associated with the model
optimizer = get_optimizer(config, model)

# Create a learning rate scheduler for an optimizer
scheduler = get_scheduler(config, optimizer)

# A loss_function for computing the loss
loss_function = get_loss_function(config, model)

# Metrics like Top 1/5 Accuracy
metrics = [get_metrics(m) for m in config.metrics]

controlflow = get_controlflow(config)

controlflow(model=model, optimizer=optimizer, loss_function=loss_function,
            metrics=metrics, scheduler=scheduler, config=config,
            dataloader_fn=generate_dataloader)
