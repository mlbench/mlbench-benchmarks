"""Training ResNet for CIFAR-10 dataset.

This implements the 1a image recognition benchmark task, see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-image-classification-resnet-cifar-10
for more details.

.. code-block:: bash
    mpirun -n 2 --oversubscribe python resnet_cifar10_mpi.py --run_id 1
"""
import logging
import os
import time

import torch.distributed as dist
import torchtext
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer

from mlbench_core.controlflow.pytorch.controlflow import (
    compute_train_batch_metrics,
    record_train_batch_stats,
    record_validation_stats,
)
from mlbench_core.controlflow.task_args import task_main
from mlbench_core.dataset.nlp.pytorch import BPTTWikiText2
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.goals import (
    task3_time_to_preplexity_goal,
    task3_time_to_preplexity_light_goal,
)
from mlbench_core.evaluation.pytorch.metrics import Perplexity
from mlbench_core.lr_scheduler.pytorch.lr import MultistepLearningRatesWithWarmup
from mlbench_core.models.pytorch.nlp import RNNLM
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer, CheckpointFreq

from .utils.utils import build_optimizer, validation_round

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
    train_epochs = 164
    batch_size = 128
    rnn_n_hidden = 1000
    rnn_n_layers = 3
    rnn_tie_weights = True
    rnn_clip = 0.25
    drop_rate = 0.1
    rnn_weight_norm = False

    bptt_len = 30
    lr = 0.001

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    optimizer_args = {
        "lr": lr,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "nesterov": False,
    }

    scheduler_args = {
        "gamma": 0.1,
        "milestones": [150, 225],
        "warmup_duration": 5,
        "warmup_init_lr": 0,
        "scaled_lr": lr * world_size,
    }

    tokenizer = get_tokenizer("spacy")
    train_set = BPTTWikiText2(
        bptt_len, train=True, tokenizer=tokenizer, root=dataset_dir
    )
    val_set = BPTTWikiText2(
        bptt_len, train=False, tokenizer=tokenizer, root=dataset_dir
    )

    vocab = train_set.get_vocab()
    train_set = partition_dataset_by_rank(train_set, rank, world_size, shuffle=False)
    val_set = partition_dataset_by_rank(val_set, rank, world_size, shuffle=False)

    num_dataloader_workers = 2
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_dataloader_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )
    n_tokens, emb_size = len(vocab), rnn_n_hidden

    model = RNNLM(
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
        weight_norm=rnn_weight_norm,
        batch_first=True,
    )

    fp_optimizer, optimizer = build_optimizer(
        model,
        world_size,
        optimizer_args=optimizer_args,
        grad_clip=rnn_clip,
        use_cuda=use_cuda,
    )
    # Create a learning rate scheduler for an optimizer
    scheduler = MultistepLearningRatesWithWarmup(optimizer, **scheduler_args)

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss(reduction="mean")

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [Perplexity()]
    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir, rank=rank, freq=CheckpointFreq.BEST
    )

    if light_target:
        goal = task3_time_to_preplexity_light_goal
    else:
        goal = task3_time_to_preplexity_goal

    tracker = Tracker(metrics, run_id, rank, goal=goal)

    dist.barrier()
    tracker.start()

    for epoch in range(0, train_epochs):

        model.train()
        tracker.train()
        hidden = model.init_hidden(batch_size)

        num_batches_per_device_train = len(train_loader)
        # configure local step.
        for batch_idx, (data, target) in enumerate(train_loader):
            tracker.batch_start()

            hidden = model.repackage_hidden(hidden)
            tracker.record_batch_load()

            # inference and get current performance.
            # Init optimizer
            fp_optimizer.zero_grad()
            tracker.record_batch_init()

            output, hidden = model(data, hidden)
            tracker.record_batch_fwd_pass()

            loss = loss_function(output, target)
            tracker.record_batch_comp_loss()

            fp_optimizer.backward_loss(loss)
            tracker.record_batch_backprop()

            updated = fp_optimizer.step(tracker=tracker)
            if updated:
                scheduler.step()

            metrics_results = compute_train_batch_metrics(
                loss.item(),
                output,
                target,
                metrics,
            )
            tracker.record_batch_comp_metrics()

            tracker.batch_end()

            record_train_batch_stats(
                batch_idx=batch_idx,
                loss=loss.item(),
                output=output,
                metric_results=metrics_results,
                tracker=tracker,
                num_batches_per_device_train=num_batches_per_device_train,
            )

        metrics_averages, loss_average = validation_round(
            val_loader,
            model=model,
            batch_size=batch_size,
            n_tokens=n_tokens,
            metrics=metrics,
            loss_function=loss_function,
            tracker=tracker,
        )

        is_best = record_validation_stats(metrics_averages, loss_average, tracker, rank)
        checkpointer.save(
            tracker,
            model,
            optimizer,
            scheduler,
            tracker.current_epoch,
            is_best,
        )
        tracker.epoch_end()

        if tracker.goal_reached:
            print("Goal Reached!")
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
    r"""Main logic."""

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
