"""Training ResNet for CIFAR-10 dataset.

This implements the 1a image recognition benchmark task, see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-image-classification-resnet-cifar-10
for more details.

.. code-block:: bash
    mpirun -n 2 --oversubscribe python resnet_cifar10_mpi.py --run_id 1
"""
import argparse
import time
import os
import logging

from mlbench_core.controlflow.pytorch import train_round, validation_round
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.dataset.nlp.pytorch import BPTTWikiText2
from mlbench_core.evaluation.pytorch.metrics import Perplexity
from mlbench_core.models.pytorch.nlp import RNNLM
from mlbench_core.optim.pytorch.optim import CentralizedSGD
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.evaluation.goals import (
    task3_time_to_preplexity_light_goal,
    task3_time_to_preplexity_goal,
)
from mlbench_core.lr_scheduler.pytorch.lr import \
    MultistepLearningRatesWithWarmup

import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torchtext.experimental.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader

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
):
    """Train loop"""
    num_parallel_workers = 2
    max_batch_per_epoch = None
    train_epochs = 164
    batch_size = 256
    rnn_n_hidden = 200
    rnn_n_layers = 2
    rnn_tie_weights = True
    rnn_clip = 0.25
    drop_rate = 0.0
    rnn_weight_norm = False
    bptt_len = 35

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tokenizer = get_tokenizer("spacy")
    train_set = BPTTWikiText2(bptt_len, train=True, tokenizer=tokenizer, root=dataset_dir)
    vocab = train_set.get_vocab()
    print(next(iter(train_set)))

    val_set = BPTTWikiText2(bptt_len, train=False, tokenizer=tokenizer, root=dataset_dir)

    train_set = partition_dataset_by_rank(train_set, rank, world_size, shuffle=False)
    val_set = partition_dataset_by_rank(val_set, rank, world_size, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
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

    n_tokens, emb_size = len(vocab), rnn_n_hidden

    model = RNNLM(
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
        weight_norm=rnn_weight_norm,
        batch_first=True
    )

    optimizer = CentralizedSGD(
        world_size=world_size,
        model=model,
        lr=0.2,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False,
    )

    # Create a learning rate scheduler for an optimizer
    scheduler = MultistepLearningRatesWithWarmup(
        optimizer,
        world_size=world_size,
        milestones=[82, 109],
        gamma=0.1,
        lr=0.1,
        warmup_duration=5,
        warmup_linear_scaling=True,
        warmup_init_lr=None,
    )

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss(reduction="mean")

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [Perplexity()]

    if light_target:
        goal = task3_time_to_preplexity_light_goal
    else:
        goal = task3_time_to_preplexity_goal

    tracker = Tracker(metrics, run_id, rank, goal=goal)

    dist.barrier()

    tracker.start()

    for epoch in range(0, train_epochs):
        train_round(
            train_loader,
            model,
            optimizer,
            loss_function,
            metrics,
            scheduler,
            "int64",
            schedule_per="epoch",
            transform_target_type=None,
            use_cuda=use_cuda,
            max_batch_per_epoch=max_batch_per_epoch,
            init_hidden=lambda: model.init_hidden(batch_size),
            package_hidden=lambda h: model.repackage_hidden(h),
            transform_parameters=lambda m: clip_grad_norm_(m.parameters(), rnn_clip),
            tracker=tracker,
        )

        validation_round(
            val_loader,
            model,
            loss_function,
            metrics,
            run_id,
            rank,
            "int64",
            transform_target_type=None,
            use_cuda=use_cuda,
            max_batch_per_epoch=max_batch_per_epoch,
            init_hidden=lambda: model.init_hidden(batch_size),
            package_hidden=lambda h: model.repackage_hidden(h),
            tracker=tracker,
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
    validation_only=False,
    gpu=False,
    light_target=False,
):
    r"""Main logic."""

    with initialize_backends(
        comm_backend="mpi",
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
    args = parser.parse_args()

    uid = "scaling"
    dataset_dir = os.path.join(args.root_dataset, "torch", "wikitext")
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
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
