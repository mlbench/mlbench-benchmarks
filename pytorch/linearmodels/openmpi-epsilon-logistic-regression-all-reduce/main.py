"""Training Logistic Regression for epsilon dataset.

This implements the Linear Learning benchmark task,
see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-linear
-learning-logistic-regression-epsilon
for more details.
"""

import argparse
import json
import os

import torch.distributed as dist
from mlbench_core.controlflow.pytorch import train_round, validation_round
from mlbench_core.controlflow.pytorch.checkpoints_evaluation import \
    CheckpointsEvaluationControlFlow
from mlbench_core.dataset.linearmodels.pytorch.epsilon import Epsilon
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.pytorch.criterion import BCELossRegularized
from mlbench_core.lr_scheduler.pytorch.lr import SQRTTimeDecayLR
from mlbench_core.models.pytorch.linear_models import LogisticRegression
from mlbench_core.optim.pytorch.optim import CentralizedSGD
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import CheckpointFreq
from mlbench_core.utils.pytorch.checkpoint import Checkpointer
from torch.utils.data import DataLoader


def train_loop(run_id, dataset_dir, ckpt_run_dir, output_dir,
               validation_only=False, use_cuda=False, light_target=False):
    r"""Main logic."""
    num_parallel_workers = 0
    max_batch_per_epoch = None
    train_epochs = 20
    batch_size = 100

    n_features = 2000
    alpha = 200
    l1_coef = 0.0000025
    l2_coef = 0.0

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = LogisticRegression(n_features)

    optimizer = CentralizedSGD(world_size=world_size, model=model, lr=0.1)

    # Create a learning rate scheduler for an optimizer
    scheduler = SQRTTimeDecayLR(optimizer, alpha)

    # A loss_function for computing the loss
    loss_function = BCELossRegularized(l1=l1_coef, l2=l2_coef, model=model)

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    metrics = []

    train_set = Epsilon(dataset_dir, train=True, download=True)
    val_set = Epsilon(dataset_dir, train=False,
                      download=False)  # No need to download again

    train_set = partition_dataset_by_rank(train_set, rank, world_size)
    val_set = partition_dataset_by_rank(val_set, rank, world_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False)

    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir,
        rank=rank,
        freq=CheckpointFreq.NONE)

    if not validation_only:

        tracker = Tracker(metrics, run_id, rank)

        dist.barrier()
        for epoch in range(0, train_epochs):
            train_round(train_loader, model, optimizer, loss_function, metrics,
                        scheduler, 'fp32', schedule_per='epoch',
                        transform_target_type=None, use_cuda=use_cuda,
                        max_batch_per_epoch=max_batch_per_epoch,
                        tracker=tracker)

            is_best = validation_round(val_loader, model, loss_function,
                                       metrics, run_id, rank, 'fp32',
                                       transform_target_type=None,
                                       use_cuda=use_cuda,
                                       max_batch_per_epoch=max_batch_per_epoch,
                                       tracker=tracker)

            checkpointer.save(tracker, model,
                              optimizer, scheduler,
                              tracker.current_epoch, is_best)

            tracker.epoch_end()
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
            dtype='fp32',
            max_batch_per_epoch=None)

        train_stats = cecf.evaluate_by_epochs(train_loader)
        with open(os.path.join(output_dir, "train_stats.json"), 'w') as f:
            json.dump(train_stats, f)

        val_stats = cecf.evaluate_by_epochs(val_loader)
        with open(os.path.join(output_dir, "val_stats.json"), 'w') as f:
            json.dump(val_stats, f)


def main(run_id, dataset_dir, ckpt_run_dir, output_dir, validation_only=False,
         gpu=False, light_target=False):
    r"""Main logic."""

    with initialize_backends(
            comm_backend='mpi',
            logging_level='INFO',
            logging_file=os.path.join(output_dir, 'mlbench.log'),
            use_cuda=gpu,
            seed=42,
            cudnn_deterministic=False,
            ckpt_run_dir=ckpt_run_dir,
            delete_existing_ckpts=not validation_only):
        train_loop(run_id, dataset_dir, ckpt_run_dir, output_dir,
                   validation_only, use_cuda=gpu, light_target=light_target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, default='1',
                        help='The id of the run')
    parser.add_argument('--root-dataset', type=str, default='/datasets',
                        help='Default root directory to dataset.')
    parser.add_argument('--root-checkpoint', type=str, default='/checkpoint',
                        help='Default root directory to checkpoint.')
    parser.add_argument('--root-output', type=str, default='/output',
                        help='Default root directory to output.')
    parser.add_argument('--validation_only', action='store_true',
                        default=False, help='Only validate from checkpoints.')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Train with GPU')
    parser.add_argument('--light', action='store_true', default=False,
                        help='Train to light target metric goal')
    args = parser.parse_args()

    uid = 'benchmark'
    dataset_dir = os.path.join(args.root_dataset, 'torch', 'epsilon')
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(args.run_id, dataset_dir, ckpt_run_dir,
         output_dir, args.validation_only), args.gpu
