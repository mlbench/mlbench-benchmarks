"""Training Logistic Regression for epsilon dataset."""

import os
import argparse
import json

from mlbench_core.controlflow.pytorch import TrainValidation
from mlbench_core.dataset.linearmodels.pytorch.dataloader import load_libsvm_lmdb, partition_dataset_by_rank
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.models.pytorch.linear_models import LogisticRegression
from mlbench_core.optim.pytorch.optim import CentralizedSGD
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import CheckpointFreq
from mlbench_core.utils.pytorch.checkpoint import Checkpointer

import torch.distributed as dist
from mlbench_core.evaluation.pytorch.criterion import BCELossRegularized
from mlbench_core.lr_scheduler.pytorch.lr import TimeDecayLR, SQRTTimeDecayLR
from torch.utils.data import DataLoader


def main(run_id, dataset_dir, ckpt_run_dir, output_dir, validation_only=False):
    r"""Main logic."""
    num_parallel_workers = 0
    use_cuda = False
    max_batch_per_epoch = None
    train_epochs = 20
    batch_size = 100
    
    n_features = 2000
    alpha = 200
    l1_coef = 0.0000025
    l2_coef = 0.0

    initialize_backends(
        comm_backend='mpi',
        logging_level='INFO',
        logging_file=os.path.join(output_dir, 'mlbench.log'),
        use_cuda=use_cuda,
        seed=42,
        cudnn_deterministic=False,
        ckpt_run_dir=ckpt_run_dir,
        delete_existing_ckpts=not validation_only)

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
    
    train_set = load_libsvm_lmdb('epsilon-train', dataset_dir)
    val_set = load_libsvm_lmdb('epsilon-train', dataset_dir)

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
        controlflow = TrainValidation(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            scheduler=scheduler,
            batch_size=batch_size,
            train_epochs=train_epochs,
            rank=rank,
            world_size=world_size,
            run_id=run_id,
            dtype='fp32',
            validate=True,
            schedule_per='epoch',
            checkpoint=checkpointer,
            transform_target_type=True,
            average_models=True,
            use_cuda=use_cuda,
            max_batch_per_epoch=max_batch_per_epoch)

        controlflow.run(
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            dataloader_train_fn=None,
            dataloader_val_fn=None,
            resume=False,
            repartition_per_epoch=False)
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
    args = parser.parse_args()

    uid = 'benchmark'
    dataset_dir = os.path.join(args.root_dataset, 'torch', 'epsilon/epsilon_train.lmdb')
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)

    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(args.run_id, dataset_dir, ckpt_run_dir,
         output_dir, args.validation_only)
