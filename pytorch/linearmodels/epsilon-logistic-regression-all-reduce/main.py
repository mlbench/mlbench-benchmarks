"""Training Logistic Regression for epsilon dataset.
This implements the Linear Learning benchmark task,
see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-linear
-learning-logistic-regression-epsilon
for more details.
Values are taken from https://arxiv.org/pdf/1705.07751.pdf
"""

import argparse
import json
import os
import time

import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from mlbench_core.dataset.linearmodels.pytorch.dataloader import LMDBDataset
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.goals import (
    task2_time_to_accuracy_goal,
    task2_time_to_accuracy_light_goal,
)
from mlbench_core.evaluation.pytorch.criterion import BCELossRegularized
from mlbench_core.evaluation.pytorch.metrics import (
    DiceCoefficient,
    F1Score,
    TopKAccuracy,
)
from mlbench_core.models.pytorch.linear_models import LogisticRegression
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
    """Main logic."""
    num_parallel_workers = 2
    max_batch_per_epoch = None
    train_epochs = 20
    batch_size = 100

    n_features = 2000

    l1_coef = 0.0
    l2_coef = 0.0000025  # Regularization 1 / train_size ( 1 / 400,000)
    dtype = "fp32"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    lr = 4
    scaled_lr = lr * min(16, world_size)

    by_layer = False
    agg_grad = False  # According to paper, we aggregate weights after update

    model = LogisticRegression(n_features)

    # A loss_function for computing the loss
    loss_function = BCELossRegularized(l1=l1_coef, l2=l2_coef, model=model)

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    optimizer = CentralizedSGD(
        world_size=world_size,
        model=model,
        lr=scaled_lr,
        use_cuda=use_cuda,
        by_layer=by_layer,
        agg_grad=agg_grad,
    )

    metrics = [
        TopKAccuracy(),  # Binary accuracy with threshold 0.5
        F1Score(),
        DiceCoefficient(),
    ]

    train_set = LMDBDataset(name="epsilon", data_type="train", root=dataset_dir)
    val_set = LMDBDataset(name="epsilon", data_type="test", root=dataset_dir)

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

    num_batches_per_device_train = len(train_loader)

    scheduler = ReduceLROnPlateau(
        optimizer.optimizer,
        factor=0.75,
        patience=0,
        verbose=True,
        threshold_mode="abs",
        threshold=0.01,
        min_lr=lr,
    )
    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.NONE
    )

    if not validation_only:
        if light_target:
            goal = task2_time_to_accuracy_light_goal()
        else:
            goal = task2_time_to_accuracy_goal()

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

                # scheduler.batch_step()
                tracker.batch_end()

                record_train_batch_stats(
                    batch_idx,
                    loss.item(),
                    output,
                    metrics_results,
                    tracker,
                    num_batches_per_device_train,
                )

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
            # Scheduler per epoch
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
    backend,
    hosts,
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
        cudnn_deterministic=False,
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
