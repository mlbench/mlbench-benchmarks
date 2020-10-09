"""Training GNMT for WMT16 Dataset

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
from torch.utils.data import DataLoader
from utils import (
    build_optimizer,
    compute_loss,
    compute_model_output,
    opt_step,
    prepare_batch,
    validation_round,
)

from mlbench_core.controlflow.pytorch.checkpoints_evaluation import (
    CheckpointsEvaluationControlFlow,
)
from mlbench_core.controlflow.pytorch.controlflow import (
    record_train_batch_stats,
    record_validation_stats,
)
from mlbench_core.dataset.nlp.pytorch import WMT16Dataset, build_collate_fn
from mlbench_core.dataset.nlp.pytorch.wmt16 import wmt16_config
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.goals import task4_time_to_bleu_goal
from mlbench_core.evaluation.pytorch.criterion import LabelSmoothing
from mlbench_core.evaluation.pytorch.metrics import BLEUScore
from mlbench_core.lr_scheduler.pytorch.lr import ExponentialWarmupMultiStepLR
from mlbench_core.models.pytorch.gnmt import GNMT, Translator
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq

try:
    import horovod.torch as hvd
except ImportError as e:
    hvd = None

logger = logging.getLogger("mlbench")


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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    train_epochs = 8
    train_min_len, train_max_len = 0, 75
    val_min_len, val_max_len = 0, 150
    math_mode = "fp16"  # One of `fp16`, `fp32` or `amp_fp16`
    lang = ("en", "de")

    # Training
    train_global_batch_size = 2048  # Global batch size
    max_bs = 128  # Max batch size for used hardware
    update_freq = int(max(1, train_global_batch_size // (max_bs * world_size)))
    train_batch_size = int(train_global_batch_size // (world_size * update_freq))
    val_batch_size = 64

    # Model attributes
    model_args = {
        "hidden_size": 1024,
        "num_layers": 4,
        "dropout": 0.2,
        "share_embedding": True,
        "fusion": True,
    }

    # Criterion
    criterion_args = {"smoothing": 0.1, "fast_xentropy": True}

    # Loss scaling
    loss_scaling = {"init_scale": 1024, "upscale_interval": 128}

    # Optimizer
    optimizer_args = {
        "lr": 2e-3,
        "grad_clip": 5.0,
    }

    # Scheduler
    scheduler_args = {
        "warmup_steps": 200,
        "remain_steps": 0.4,
        "decay_interval": 0.05,
        "decay_steps": 4,
        "decay_factor": 0.5,
    }

    # Translator
    translator_args = {
        "beam_size": 5,
        "len_norm_factor": 0.6,
        "cov_penalty_factor": 0.1,
        "len_norm_const": 5.0,
        "max_seq_len": 150,
    }

    # Build train/val datsets
    train_set = WMT16Dataset(
        dataset_dir,
        math_precision=math_mode,
        lang=lang,
        train=True,
        download=True,
        preprocessed=True,
        min_len=train_min_len,
        max_len=train_max_len,
    )
    train_set.prepare()
    val_set = WMT16Dataset(
        dataset_dir,
        math_precision=math_mode,
        lang=lang,
        validation=True,
        download=False,
        min_len=val_min_len,
        max_len=val_max_len,
        sort=True,
    )

    tokenizer = train_set.tokenizer

    # Build model
    model = GNMT(vocab_size=train_set.vocab_size, **model_args)

    # Build loss function
    criterion = LabelSmoothing(padding_idx=wmt16_config.PAD, **criterion_args)

    # Bilingual Evaluation Understudy Score
    metrics = [BLEUScore()]

    # Partition data
    train_set = partition_dataset_by_rank(train_set, rank, world_size)
    val_set = partition_dataset_by_rank(val_set, rank, world_size)

    collate_fn = build_collate_fn(sort=True)
    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
    )

    validate_every = update_freq * round(
        len(train_loader) * 0.30 / update_freq
    )  # Validate every 30%

    # Build optimizer & scheduler
    total_train_iters = (len(train_loader) // update_freq) * train_epochs

    print("Number of batches per epoch {}".format(len(train_loader)))
    print("Train iterations per epoch {}".format(total_train_iters / train_epochs))

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    use_horovod = (
        math_mode == "fp16" or math_mode == "amp_fp16"
    ) and dist.get_backend() == dist.Backend.MPI

    if use_horovod:
        hvd.init()
        logger.info("Using horovod rank={}".format(hvd.rank()))
        tensor = torch.tensor([1])
        res = hvd.allreduce(tensor, op=hvd.Sum)
        assert res[0] == world_size

    fp_optimizer, optimizer, model = build_optimizer(
        model=model,
        math=math_mode,
        loss_scaling=loss_scaling,
        use_cuda=use_cuda,
        world_size=world_size,
        use_horovod=use_horovod,
        **optimizer_args
    )

    # Create a learning rate scheduler for an optimizer
    scheduler = ExponentialWarmupMultiStepLR(
        optimizer, total_train_iters, **scheduler_args
    )

    # Translator
    translator = Translator(model=model, trg_tokenizer=tokenizer, **translator_args)

    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.BEST
    )

    if not validation_only:

        if light_target:
            goal = task4_time_to_bleu_goal(20)
        else:
            goal = task4_time_to_bleu_goal(24)

        num_batches_per_device_train = len(train_loader)
        tracker = Tracker(metrics, run_id, rank, goal=goal)

        dist.barrier()
        tracker.start()

        for epoch in range(0, train_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model.train()
            tracker.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                tracker.batch_start()
                data, target = prepare_batch(data, target, use_cuda=use_cuda)
                tracker.record_batch_load()

                is_last = batch_idx == len(train_loader)
                update = (batch_idx % update_freq) == update_freq - 1
                init = (batch_idx % update_freq) == 0

                # Clear gradients in the optimizer.
                if init:
                    fp_optimizer.zero_grad()
                    tracker.record_batch_init()

                # Compute the output
                output = compute_model_output(model, data, target)
                tracker.record_batch_fwd_pass()

                # Compute the loss
                loss, loss_per_token = compute_loss(
                    data, target, output, criterion, update_freq
                )
                tracker.record_batch_comp_loss()
                # Backprop
                fp_optimizer.backward_loss(loss)
                tracker.record_batch_backprop()

                # Opt step
                if update or is_last:
                    updated = opt_step(
                        fp_optimizer,
                        update_freq,
                        math_mode,
                        world_size,
                        tracker=tracker,
                    )
                    # Learning rate scheduler
                    if updated:
                        scheduler.step()

                tracker.batch_end()

                record_train_batch_stats(
                    batch_idx=batch_idx,
                    loss=loss_per_token,
                    output=target[0],  # Use target just for the size
                    metric_results={},
                    tracker=tracker,
                    num_batches_per_device_train=num_batches_per_device_train,
                )

                # Validation during training
                if (batch_idx + 1) % validate_every == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    metrics_values, loss = validation_round(
                        val_loader,
                        metrics,
                        model,
                        criterion,
                        update_freq,
                        translator,
                        tracker=tracker,
                        use_cuda=use_cuda,
                    )

                    record_validation_stats(metrics_values, loss, tracker, rank)
                    if tracker.goal_reached:
                        break

                    model.train()
                    tracker.train()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            metrics_values, loss = validation_round(
                val_loader,
                metrics,
                model,
                criterion,
                update_freq,
                translator,
                use_cuda=use_cuda,
            )

            is_best = record_validation_stats(metrics_values, loss, tracker, rank)

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
            loss_function=criterion,
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

    uid = "allreduce"

    dataset_dir = os.path.join(args.root_dataset, "torch", "wmt16")
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
        rank=args.rank,
        backend=args.backend,
        hosts=args.hosts,
        validation_only=args.validation_only,
        gpu=args.gpu,
        light_target=args.light,
    )
