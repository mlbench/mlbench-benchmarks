"""Training ResNet for CIFAR-10 dataset.

This implements the 1a image recognition benchmark task,
see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-image
-classification-resnet-cifar-10
for more details.

.. code-block:: bash
    mpirun -n 2 --oversubscribe python resnet_cifar10_mpi.py --run_id 1
"""
import argparse
import json
import math
import os
import time

import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

from mlbench_core.controlflow.pytorch import (
    compute_train_batch_metrics,
    prepare_batch,
    record_train_batch_stats,
    record_validation_stats,
    validation_round,
)
from mlbench_core.controlflow.pytorch.checkpoints_evaluation import (
    CheckpointsEvaluationControlFlow,
)
from mlbench_core.dataset.imagerecognition.pytorch import CIFAR10V1
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.goals import (
    task1_time_to_accuracy_goal,
    task1_time_to_accuracy_light_goal,
)
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.lr_scheduler.pytorch.lr import ReduceLROnPlateauWithWarmup
from mlbench_core.models.pytorch.resnet import ResNetCIFAR
from mlbench_core.optim.pytorch.centralized import CentralizedSGD
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq
from mlbench_core.utils.task_args import task_main


def train_loop(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    validation_only=False,
    use_cuda=False,
    light_target=False,
):
    """Train loop"""
    num_parallel_workers = 2
    max_batch_per_epoch = None
    train_epochs = 164
    batch_size = 128
    dtype = "fp32"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # LR = 0.1 / 256 / sample
    lr = 0.02
    scaled_lr = lr * world_size
    by_layer = False

    # Create Model
    model = ResNetCIFAR(resnet_size=20, bottleneck=False, num_classes=10, version=1)

    # Create optimizer
    optimizer = CentralizedSGD(
        world_size=world_size,
        model=model,
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
        use_cuda=use_cuda,
        by_layer=by_layer,
    )

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [TopKAccuracy(topk=1), TopKAccuracy(topk=5)]

    train_set = CIFAR10V1(dataset_dir, train=True, download=True)
    val_set = CIFAR10V1(dataset_dir, train=False, download=True)

    # Create train/validation sets and loaders
    train_set = partition_dataset_by_rank(train_set, rank, world_size)
    val_set = partition_dataset_by_rank(val_set, rank, world_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False,
    )

    # Create a learning rate scheduler for an optimizer
    scheduler = ReduceLROnPlateauWithWarmup(
        optimizer.optimizer,
        warmup_init_lr=lr,
        scaled_lr=scaled_lr,
        warmup_epochs=int(math.log(world_size, 2)),  # Adaptive warmup period
        factor=0.5,
        threshold_mode="abs",
        threshold=0.01,
        patience=1,
        verbose=True,
        min_lr=lr,
    )

    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.NONE
    )

    if not validation_only:
        if light_target:
            goal = task1_time_to_accuracy_light_goal()
        else:
            goal = task1_time_to_accuracy_goal()

        num_batches_per_device_train = len(train_loader)

        tracker = Tracker(metrics, run_id, rank, goal=goal)

        dist.barrier()

        tracker.start()

        for epoch in range(0, train_epochs):
            # Set tracker and model in training mode
            model.train()
            tracker.train()

            for batch_idx, (data, target) in enumerate(train_loader):
                tracker.batch_start()
                data, target = prepare_batch(
                    data,
                    target,
                    dtype=dtype,
                    transform_target_dtype=False,
                    use_cuda=use_cuda,
                )
                tracker.record_batch_load()

                # Clear gradients in the optimizer.
                optimizer.zero_grad()
                tracker.record_batch_init()

                # Compute the output
                output = model(data)
                tracker.record_batch_fwd_pass()

                # Compute the loss
                loss = loss_function(output, target)
                tracker.record_batch_comp_loss()

                # Backprop
                loss.backward()
                tracker.record_batch_backprop()

                # Aggregate gradients/parameters from all workers and apply updates to model
                optimizer.step(tracker=tracker)

                metrics_results = compute_train_batch_metrics(
                    output,
                    target,
                    metrics,
                )
                tracker.record_batch_comp_metrics()
                tracker.batch_end()

                record_train_batch_stats(
                    batch_idx,
                    loss.item(),
                    output,
                    metrics_results,
                    tracker,
                    num_batches_per_device_train,
                )

            # Scheduler per epoch
            tracker.epoch_end()

            # Perform validation and gather results
            metrics_values, loss = validation_round(
                val_loader,
                model=model,
                loss_function=loss_function,
                metrics=metrics,
                dtype=dtype,
                tracker=tracker,
                transform_target_type=False,
                use_cuda=use_cuda,
                max_batches=max_batch_per_epoch,
            )
            scheduler.step(loss)

            # Record validation stats
            is_best = record_validation_stats(
                metrics_values=metrics_values, loss=loss, tracker=tracker, rank=rank
            )

            checkpointer.save(tracker, model, optimizer, scheduler, is_best)

            if tracker.goal_reached:
                print("Goal Reached!")
                dist.barrier()
                time.sleep(10)
                return
    else:
        cecf = CheckpointsEvaluationControlFlow(
            ckpt_dir=ckpt_run_dir,
            rank=rank,
            world_size=world_size,
            checkpointer=checkpointer,
            model=model,
            epochs=train_epochs,
            loss_function=loss_function,
            metrics=metrics,
            use_cuda=use_cuda,
            dtype="fp32",
            max_batch_per_epoch=None,
        )

        train_stats = cecf.evaluate_by_epochs(train_loader)
        with open(os.path.join(output_dir, "train_stats.json"), "w") as f:
            json.dump(train_stats, f)

        val_stats = cecf.evaluate_by_epochs(val_loader)
        with open(os.path.join(output_dir, "val_stats.json"), "w") as f:
            json.dump(val_stats, f)


def main(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    rank,
    hosts,
    backend,
    validation_only=False,
    gpu=False,
    light_target=False,
):
    r"""Main logic."""

    with initialize_backends(
        comm_backend=backend,
        hosts=hosts,
        rank=rank,
        logging_level="INFO",
        logging_file=os.path.join(output_dir, "mlbench.log"),
        use_cuda=gpu,
        seed=42,
        cudnn_deterministic=True,
        ckpt_run_dir=ckpt_run_dir,
        delete_existing_ckpts=not validation_only,
    ):
        train_loop(
            run_id,
            dataset_dir,
            ckpt_run_dir,
            output_dir,
            validation_only,
            use_cuda=gpu,
            light_target=light_target,
        )


if __name__ == "__main__":
    task_main(main)
