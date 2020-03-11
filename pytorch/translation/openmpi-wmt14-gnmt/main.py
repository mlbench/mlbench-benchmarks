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
import logging
import os
import time

import torch
import torch.distributed as dist
from mlbench_core.controlflow.pytorch.checkpoints_evaluation import \
    CheckpointsEvaluationControlFlow
from mlbench_core.controlflow.pytorch.gnmt import GNMTTrainer
from mlbench_core.dataset.translation.pytorch.dataloader import WMT14Dataset
from mlbench_core.evaluation.goals import task1_time_to_accuracy_goal, \
    task1_time_to_accuracy_light_goal
from mlbench_core.evaluation.pytorch.inference import Translator
from mlbench_core.evaluation.pytorch.metrics import BLEUScore
from mlbench_core.evaluation.pytorch.utils import build_criterion
from mlbench_core.lr_scheduler.pytorch.lr import \
    ExponentialWarmupMultiStepLR
from mlbench_core.models.pytorch.gnmt import GNMT
from mlbench_core.optim.pytorch.utils import build_fp_optimizer
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import CheckpointFreq, Checkpointer

logger = logging.getLogger('mlbench')


def train_loop(run_id, dataset_dir, ckpt_run_dir, output_dir,
               validation_only=False, use_cuda=False, light_target=False):
    """Train loop"""
    num_parallel_workers = 2
    max_batch_per_epoch = None
    train_epochs = 10

    # Dataset attributes
    train_min_len, train_max_len = 0, 50
    val_min_len, val_max_len = 0, 125
    math_mode = "fp32"
    batch_first = False
    include_lengths = True
    max_size = None

    # Model attributes
    hidden_size = 1024
    num_layers = 4
    dropout = 0.2
    share_embedding = True
    smoothing = 0.1

    # Training
    train_batch_size = 128
    train_iter_size = 1
    val_batch_size = 64
    grad_clip = 5.0

    # Optimizer
    opt_config = {'optimizer': "Adam", 'lr': 2.00e-3}

    # Loss
    loss_scaling = {
        'init_scale': 8192,
        'upscale_interval': 128
    }

    # Scheduler
    scheduler_config = {'warmup_steps': 200,
                        'remain_steps': 0.666,
                        'decay_interval': None,
                        'decay_steps': 4,
                        'decay_factor': 0.5}

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build train/val datsets
    train_set = WMT14Dataset(dataset_dir,
                             train=True,
                             download=True,
                             include_lengths=include_lengths,
                             math_precision=math_mode,
                             batch_first=batch_first,
                             min_len=train_min_len,
                             max_len=train_max_len,
                             max_size=max_size)

    val_set = WMT14Dataset(dataset_dir,
                           validation=True,
                           download=False,
                           include_lengths=include_lengths,
                           math_precision=math_mode,
                           batch_first=batch_first,
                           min_len=val_min_len,
                           max_len=val_max_len,
                           max_size=max_size)

    # train_set = partition_dataset_by_rank(train_set, rank, world_size)
    # val_set = partition_dataset_by_rank(val_set, rank, world_size)

    # Build model
    model = GNMT(vocab_size=train_set.vocab_size,
                 padding_idx=train_set.get_special_token_idx("PAD"),
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 dropout=dropout,
                 batch_first=batch_first,
                 share_embedding=share_embedding,
                 )

    # Build optimizer
    fp_optimizer, optimizer, model = build_fp_optimizer(model=model, math=math_mode,
                                                        opt_config=opt_config, grad_clip=grad_clip, loss_scaling=loss_scaling)

    # Build loss function
    loss_function = build_criterion(train_set.get_special_token_idx("PAD"), smoothing)

    total_train_iters = len(train_set) // train_iter_size * train_epochs

    # Create a learning rate scheduler for an optimizer
    scheduler = ExponentialWarmupMultiStepLR(optimizer,
                                             total_train_iters,
                                             **scheduler_config)

    # Translator
    translator_config = {
        "model": model,
        "trg_tokenizer": train_set.fields['trg'],
        "BOS_idx": train_set.get_special_token_idx("BOS"),
        "EOS_idx": train_set.get_special_token_idx("EOS"),
    }
    translator = Translator(**translator_config)

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [BLEUScore()]

    train_loader = train_set.get_loader(batch_size=train_batch_size,
                                        shuffle=False,
                                        device=torch.device(
                                            'cuda' if use_cuda else 'cpu'))
    val_loader = val_set.get_loader(batch_size=val_batch_size,
                                    shuffle=False,
                                    device=torch.device(
                                        'cuda' if use_cuda else 'cpu'))

    trainer = GNMTTrainer(model=model,
                          criterion=loss_function,
                          fp_optimizer=fp_optimizer,
                          scheduler=scheduler,
                          translator=translator,
                          rank=rank,
                          schedule_per='batch',
                          tracker=None,
                          metrics=metrics,
                          iter_size=train_iter_size)
    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir,
        rank=rank,
        freq=CheckpointFreq.NONE)

    if not validation_only:
        if light_target:
            goal = task1_time_to_accuracy_light_goal()
        else:
            goal = task1_time_to_accuracy_goal()

        tracker = Tracker(metrics, run_id, rank, goal=goal)

        trainer.set_tracker(tracker)
        dist.barrier()

        tracker.start()

        for epoch in range(0, train_epochs):
            trainer.train_round(train_loader)
            # train_round(train_loader, model, optimizer, loss_function, metrics,
            #             scheduler, 'fp32', schedule_per='epoch',
            #             transform_target_type=None, use_cuda=use_cuda,
            #             max_batch_per_epoch=max_batch_per_epoch,
            #             tracker=tracker)

            # is_best = validation_round(val_loader, model, loss_function,
            #                            metrics, run_id, rank, 'fp32',
            #                            transform_target_type=None,
            #                            use_cuda=use_cuda,
            #                            max_batch_per_epoch=max_batch_per_epoch,
            #                            tracker=tracker)

            is_best = trainer.validation_round(val_loader)
            checkpointer.save(tracker, model,
                              optimizer, scheduler,
                              tracker.current_epoch, is_best)

            tracker.epoch_end()

            if tracker.goal_reached:
                print("Goal Reached!")
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

    uid = 'scaling'
    dataset_dir = os.path.join(args.root_dataset, 'torch', 'wmt14')
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(args.run_id, dataset_dir, ckpt_run_dir,
         output_dir, validation_only=args.validation_only, gpu=args.gpu,
         light_target=args.light)
