"""Training AWD-LSTM for Wikitext2 DataSet

This implements the Language Modeling task 3a,
see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#task-3-language-modelling
for more details.

Model and training taken from https://github.com/salesforce/awd-lstm-lm
"""
import logging
import math
import os
import time

import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import ASGD, SGD
from utils import repackage_hidden, set_sequence_lengths, validation_round

from mlbench_core.controlflow.pytorch.controlflow import (
    compute_train_batch_metrics,
    record_train_batch_stats,
    record_validation_stats,
)
from mlbench_core.dataset.nlp.pytorch import Wikitext2Dataset
from mlbench_core.evaluation.goals import task3_time_to_perplexity_goal
from mlbench_core.evaluation.pytorch.metrics import Perplexity
from mlbench_core.lr_scheduler.pytorch.lr import LRLinearWarmUp
from mlbench_core.models.pytorch.language_models import LSTMLanguageModel
from mlbench_core.optim.pytorch.centralized import CustomCentralizedOptimizer
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq
from mlbench_core.utils.task_args import task_main

LOG_EVERY_N_BATCHES = 25
logger = logging.getLogger("mlbench")


def train_loop(
    run_id,
    dataset_dir,
    ckpt_run_dir,
    output_dir,
    validation_only=False,
    use_cuda=False,
    light_target=False,
    seed=42,
):
    """Train loop"""
    train_epochs = 750

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Using batch scaling
    train_batch_size = 80
    train_global_batch_size = train_batch_size * world_size

    val_batch_size = 10
    # Define the batch sizes here

    # Dataset arguments
    bptt = 70
    min_seq_len = 5

    # Model Arguments
    model_args = {
        "ninp": 400,
        "nhid": 1150,
        "nlayers": 3,
        "dropout": 0.4,
        "dropouth": 0.2,
        "dropouti": 0.65,
        "dropoute": 0.1,
        "wdrop": 0.5,
        "tie_weights": True,
    }

    # Optimizer args
    lr = 30
    scaled_lr = lr * math.sqrt(world_size)
    warmup_epochs = 5 * world_size
    weight_decay = 1.2e-6
    grad_clip = 0.25
    alpha = 2
    beta = 1
    nonmono = 5

    # Load train/valid
    train_set = Wikitext2Dataset(
        dataset_dir, bptt=bptt, train=True, min_seq_len=min_seq_len
    )
    val_set = Wikitext2Dataset(
        dataset_dir, bptt=bptt, valid=True, min_seq_len=min_seq_len
    )
    ntokens = len(train_set.dictionary)

    # Generate batches
    train_set.generate_batches(
        global_bsz=train_global_batch_size, worker_bsz=train_batch_size, rank=rank
    )

    val_set.generate_batches(val_batch_size)
    val_set.generate_sequence_lengths()

    logger.info("Built dictionary of {} tokens".format(ntokens))

    model = LSTMLanguageModel(ntokens, **model_args)
    criterion = CrossEntropyLoss(reduction="mean")
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer = SGD(model.parameters(), lr=scaled_lr, weight_decay=weight_decay)
    c_optimizer = CustomCentralizedOptimizer(
        model=model,
        optimizer=optimizer,
        use_cuda=use_cuda,
        agg_grad=True,
        grad_clip=grad_clip,
        world_size=world_size,
    )

    scheduler = LRLinearWarmUp(
        optimizer,
        init_lr=lr / world_size,
        scaled_lr=scaled_lr,
        warmup_duration=warmup_epochs,
    )
    metrics = [Perplexity()]

    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.NONE
    )

    if light_target:
        goal = task3_time_to_perplexity_goal(90)
    else:
        goal = task3_time_to_perplexity_goal(70)

    tracker = Tracker(metrics, run_id, rank, goal=goal, minimize=True)

    dist.barrier()
    tracker.start()

    val_losses = []
    for epoch in range(0, train_epochs):
        model.train()
        tracker.train()

        # Init hidden state
        hidden = model.init_hidden(train_batch_size)

        # Set random sequence lengths for epoch
        set_sequence_lengths(train_set, random=True)

        num_batches_per_device_train = train_set.num_batches()

        for batch_idx in range(num_batches_per_device_train):
            tracker.batch_start()
            data, targets = train_set.get_batch(batch_idx, cuda=use_cuda)

            seq_len = data.size(0)
            lr_original = optimizer.param_groups[0]["lr"]
            batch_lr = lr_original * seq_len / bptt
            optimizer.param_groups[0]["lr"] = batch_lr

            hidden = repackage_hidden(hidden)
            c_optimizer.zero_grad()
            tracker.record_batch_init()

            output, hidden, raw_outputs, outputs = model(data, hidden, return_h=True)
            tracker.record_batch_fwd_pass()

            loss = criterion(output, targets)
            # Activation regularization
            loss = loss + sum(
                alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in outputs[-1:]
            )
            # Temporal Activation Regularization (slowness)
            loss = loss + sum(
                beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean()
                for rnn_h in raw_outputs[-1:]
            )
            tracker.record_batch_comp_loss()

            loss.backward()
            tracker.record_batch_backprop()

            c_optimizer.step(tracker=tracker)

            optimizer.param_groups[0]["lr"] = lr_original
            metrics_results = compute_train_batch_metrics(
                output,
                targets,
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
        tracker.epoch_end()

        # Still in regular SGD
        if type(c_optimizer.optimizer) == SGD:
            metrics_values, loss = validation_round(
                val_set,
                model=model,
                batch_size=val_batch_size,
                metrics=metrics,
                loss_function=criterion,
                tracker=tracker,
                use_cuda=use_cuda,
            )
            scheduler.step()
            logger.info("Using LR={}".format(scheduler.get_last_lr()))
            if len(val_losses) > nonmono and loss > min(val_losses[:-nonmono]):
                logger.info("Switching optimizer to ASGD")
                optimizer = ASGD(
                    params=model.parameters(),
                    lr=scheduler.get_last_lr()[0],
                    lambd=0.0,
                    t0=0,
                    weight_decay=weight_decay,
                )
                c_optimizer.optimizer = optimizer

        # Switched to ASGD, no scheduling
        else:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]["ax"].clone()

            metrics_values, loss = validation_round(
                val_set,
                model=model,
                batch_size=val_batch_size,
                metrics=metrics,
                loss_function=criterion,
                tracker=tracker,
                use_cuda=use_cuda,
            )

            for prm in model.parameters():
                prm.data = tmp[prm].clone()
        val_losses.append(loss)

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
    """Main logic."""

    with initialize_backends(
        comm_backend=backend,
        hosts=hosts,
        rank=rank,
        logging_level="INFO",
        logging_file=os.path.join(output_dir, "mlbench.log"),
        use_cuda=gpu,
        seed=43,
        cudnn_deterministic=False,
        ckpt_run_dir=ckpt_run_dir,
        delete_existing_ckpts=not validation_only,
    ):
        train_loop(
            run_id,
            dataset_dir,
            ckpt_run_dir,
            output_dir,
            validation_only=validation_only,
            use_cuda=gpu,
            light_target=light_target,
            seed=43,
        )


if __name__ == "__main__":
    task_main(main)
