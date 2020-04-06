"""Training GNMT for WMT14 Dataset

This implements the machine translation benchmark tasks,
# TODO add link to docs
"""
import argparse
import json
import logging
import os
import time

import torch
import torch.distributed as dist
import torchtext
from apex import amp
from mlbench_core.controlflow.pytorch.checkpoints_evaluation import (
    CheckpointsEvaluationControlFlow,
)
from mlbench_core.controlflow.pytorch.gnmt import GNMTTrainer
from mlbench_core.dataset.translation.pytorch import WMT14Dataset, config
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.pytorch.criterion import LabelSmoothing
from mlbench_core.evaluation.pytorch.inference import Translator
from mlbench_core.evaluation.pytorch.metrics import BLEUScore
from mlbench_core.lr_scheduler.pytorch.lr import ExponentialWarmupMultiStepLR
from mlbench_core.models.pytorch.gnmt import GNMT
from mlbench_core.optim.pytorch import FP32Optimizer, AMPOptimizer, Adam, CentralizedAdam
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import CheckpointFreq, Checkpointer
from torch import nn
from mlbench_core.evaluation.goals import task4_time_to_bleu_goal

logger = logging.getLogger("mlbench")


def set_iter_size(global_bs, train_bs):
    """
    Automatically set train_iter_size based on train_global_batch_size,
    world_size and per-worker train_batch_size

    """
    world_size = dist.get_world_size()
    assert global_bs % (train_bs * world_size) == 0
    train_iter_size = global_bs // (train_bs * world_size)
    logger.info(f'Global batch size was set, '
                f'Setting train_iter_size to {train_iter_size}')
    return train_iter_size


def build_optimizer(
        model, math, optimizer, grad_clip, loss_scaling
):
    if math == "fp32":
        fp_optimizer = FP32Optimizer(
            model=model, optimizer=optimizer, grad_clip=grad_clip
        )

    elif math == "fp16":
        model, optimizer = amp.initialize(
            model,
            optimizer,
            cast_model_outputs=torch.float16,
            keep_batchnorm_fp32=False,
            opt_level="O2",
        )

        fp_optimizer = AMPOptimizer(
            model,
            optimizer,
            grad_clip=grad_clip,
            loss_scale=loss_scaling["init_scale"],
            dls_upscale_interval=loss_scaling["upscale_interval"],
        )
    else:
        return NotImplementedError()

    return fp_optimizer, model


def build_criterion(padding_idx, smoothing):
    if smoothing == 0.0:
        criterion = nn.CrossEntropyLoss(ignore_index=padding_idx, size_average=False)
    else:
        criterion = LabelSmoothing(padding_idx, smoothing)

    return criterion


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
    train_epochs = 6

    # Dataset attributes
    train_min_len, train_max_len = 0, 50
    val_min_len, val_max_len = 0, 150
    math_mode = "fp16"  # One of `fp16`, `fp32`
    batch_first = False
    include_lengths = True
    lang = {"src": "en", "trg": "de"}

    # Model attributes
    hidden_size = 1024
    num_layers = 4
    dropout = 0.2
    share_embedding = True
    smoothing = 0.1

    # Training
    train_batch_size = 128
    train_global_batch_size = 1024 if dist.get_world_size() <= 8 else 2048
    train_iter_size = set_iter_size(train_global_batch_size, train_batch_size)
    val_batch_size = 64
    validate_every = 2000

    assert (validate_every % train_iter_size) == 0
    # Translator
    beam_size = 5
    len_norm_factor = 0.6
    cov_penalty_factor = 0.1
    len_norm_const = 5.0
    max_seq_len = 150

    # Optimizer
    lr = 2.00e-3
    grad_clip = 5.0

    # Loss
    loss_scaling = {"init_scale": 8192, "upscale_interval": 128}

    # Scheduler
    scheduler_config = {
        "warmup_steps": 200,
        "remain_steps": 0.666,
        "decay_interval": None,
        "decay_steps": 4,
        "decay_factor": 0.5,
    }

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build train/val datsets
    train_set = WMT14Dataset(
        dataset_dir,
        batch_first=batch_first,
        include_lengths=include_lengths,
        math_precision=math_mode,
        lang=lang,
        train=True,
        download=True,
        lazy=True,
        min_len=train_min_len,
        max_len=train_max_len,
    )

    val_set = WMT14Dataset(
        dataset_dir,
        batch_first=batch_first,
        include_lengths=include_lengths,
        math_precision=math_mode,
        lang=lang,
        validation=True,
        download=False,
        min_len=val_min_len,
        max_len=val_max_len,
    )
    tokenizer = train_set.fields["trg"]

    # Build model
    model = GNMT(
        vocab_size=train_set.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        batch_first=batch_first,
        share_embedding=share_embedding,
    )

    # Build loss function
    loss_function = build_criterion(config.PAD, smoothing)

    # Bilingual Evaluation Understudy Score
    metrics = [BLEUScore()]

    # Partition data
    train_set = partition_dataset_by_rank(train_set, rank, world_size)

    # Get data loaders
    train_loader = torchtext.data.BucketIterator(
        dataset=train_set.data,
        batch_size=train_batch_size,
        shuffle=False,
        sort_within_batch=True,
        device=torch.device("cuda" if use_cuda else "cpu"),
        sort_key=lambda x: len(x.src),
    )

    val_loader = torchtext.data.BucketIterator(
        dataset=val_set,
        batch_size=val_batch_size,
        shuffle=False,
        sort_within_batch=True,
        device=torch.device("cuda" if use_cuda else "cpu"),
        sort_key=lambda x: len(x.src),
    )

    # Build optimizer & scheduler
    total_train_iters = (len(train_loader) // train_iter_size) * train_epochs

    logger.info("Number of batches per epoch {}".format(len(train_loader)))
    logger.info("Train iterations per epoch {}".format(total_train_iters / train_epochs))

    # optimizer = Adam(params=model.parameters(), lr=lr)

    optimizer = CentralizedAdam(world_size=world_size,
                                model=model,
                                lr=lr)
    # Create a learning rate scheduler for an optimizer
    scheduler = ExponentialWarmupMultiStepLR(optimizer,
                                             total_train_iters,
                                             **scheduler_config)

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    fp_optimizer, model = build_optimizer(
        model=model,
        math=math_mode,
        optimizer=optimizer,
        grad_clip=grad_clip,
        loss_scaling=loss_scaling,
    )

    # Translator
    translator = Translator(
        model=model,
        trg_tokenizer=tokenizer,
        beam_size=beam_size,
        len_norm_factor=len_norm_factor,
        len_norm_const=len_norm_const,
        cov_penalty_factor=cov_penalty_factor,
        max_seq_len=max_seq_len,
    )

    # Trainer
    trainer = GNMTTrainer(
        model=model,
        criterion=loss_function,
        fp_optimizer=fp_optimizer,
        scheduler=scheduler,
        translator=translator,
        rank=rank,
        schedule_per="batch",
        tracker=None,
        metrics=metrics,
        iter_size=train_iter_size,
    )
    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.BEST
    )

    if not validation_only:

        if light_target:
            goal = task4_time_to_bleu_goal(20)
        else:
            goal = task4_time_to_bleu_goal(24)

        tracker = Tracker(metrics, run_id, rank, goal=goal)

        trainer.set_tracker(tracker)
        dist.barrier()
        tracker.start()

        for epoch in range(0, train_epochs):
            trainer.train_round(train_loader, val_loader=val_loader, bleu_score=True, validate_every=validate_every)

            is_best = trainer.validation_round(val_loader)
            checkpointer.save(
                tracker,
                model,
                fp_optimizer.optimizer,
                scheduler,
                tracker.current_epoch,
                is_best,
            )

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
        validation_only=False,
        gpu=False,
        light_target=False,
):
    r"""Main logic."""
    with initialize_backends(
            comm_backend="nccl",
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
    parser.add_argument(
        "--hosts",
        type=str
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0
    )
    args = parser.parse_args()

    uid = "allreduce"
    hosts = args.hosts.split(',')
    os.environ["MASTER_ADDR"] = hosts[0]
    os.environ['MASTER_PORT'] = '29500'
    os.environ["RANK"] = str(args.rank)
    os.environ["WORLD_SIZE"] = str(len(hosts))

    dataset_dir = os.path.join(args.root_dataset, "torch", "wmt14")
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
        validation_only=args.validation_only,
        gpu=args.gpu,
        light_target=args.light,
    )
