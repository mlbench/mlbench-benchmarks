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
from mlbench_core.lr_scheduler.pytorch.lr import MultistepLearningRatesWithWarmup
from mlbench_core.models.pytorch.linear_models import LogisticRegression
from mlbench_core.optim.pytorch.optim import CentralizedSGD
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq


def train_loop(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    validation_only=False,
    use_cuda=False,
    light_target=False,
):
    r"""Main logic."""
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

    lr = 2
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
        lr=lr,
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

    # Create a learning rate scheduler for an optimizer
    # Milestones for reducing LR
    milestones = [6 * num_batches_per_device_train, 12 * num_batches_per_device_train]
    scheduler = MultistepLearningRatesWithWarmup(
        optimizer,
        world_size=world_size,
        gamma=0.5,
        milestones=milestones,
        lr=lr,
        warmup_duration=num_batches_per_device_train,
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
                    loss.item(), output, target, metrics,
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
                scheduler.step()
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

            # Record validation stats
            is_best = record_validation_stats(
                metrics_values=metrics_values, loss=loss, tracker=tracker, rank=rank
            )

            checkpointer.save(
                tracker, model, optimizer, scheduler, tracker.current_epoch, is_best
            )

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
    parser = argparse.ArgumentParser(description="Process run parameters")
    parser.add_argument("--run_id", type=str, default="1", help="The id of the run")
    parser.add_argument(
        "--root-dataset",
        type=str,
        default="/datasets",
        help="Default root directory to dataset.",
    )
    parser.add_argument(
        "--root-checkpoint",
        type=str,
        default="/checkpoint",
        help="Default root directory to checkpoint.",
    )
    parser.add_argument(
        "--root-output",
        type=str,
        default="/output",
        help="Default root directory to output.",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        default=False,
        help="Only validate from checkpoints.",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Train with GPU"
    )
    parser.add_argument(
        "--light",
        action="store_true",
        default=False,
        help="Train to light target metric goal",
    )
    parser.add_argument("--rank", type=int, default=1, help="The rank of the process")
    parser.add_argument(
        "--backend", type=str, default="mpi", help="PyTorch distributed backend"
    )
    parser.add_argument("--hosts", type=str, help="The list of hosts")

    args = parser.parse_args()

    uid = "benchmark"
    dataset_dir = os.path.join(args.root_dataset, "torch", "epsilon")
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(
        args.run_id,
        dataset_dir,
        ckpt_run_dir,
        output_dir,
        args.rank,
        args.backend,
        args.hosts,
        validation_only=args.validation_only,
        gpu=args.gpu,
        light_target=args.light,
    )
