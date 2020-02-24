"""Training ResNet for CIFAR-10 dataset.

This implements the 1a image recognition benchmark task, see https://mlbench.readthedocs.io/en/latest/benchmark-tasks.html#a-image-classification-resnet-cifar-10
for more details.

.. code-block:: bash
    mpirun -n 2 --oversubscribe python resnet_cifar10_mpi.py --run_id 1
"""
import argparse
import json
import os
import logging

from mlbench_core.controlflow.pytorch import train_round, validation_round
from mlbench_core.dataset.nlp.pytorch import Wikitext2
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.pytorch.metrics import Perplexity
from mlbench_core.lr_scheduler.pytorch.lr import MultistepLearningRatesWithWarmup
from mlbench_core.models.pytorch.nlp import RNNLM
from mlbench_core.optim.pytorch.optim import CentralizedSGD
from mlbench_core.utils import Tracker, AverageMeter
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.distributed import global_average
from mlbench_core.evaluation.goals import task3_time_to_preplexity_light_goal, task3_time_to_preplexity_goal

import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
import torchtext

LOG_EVERY_N_BATCHES = 25
logger = logging.getLogger('mlbench')


def train_loop(run_id, dataset_dir, ckpt_run_dir, output_dir,
               validation_only=False, use_cuda=False, light_target=False):
    """Train loop"""
    num_parallel_workers = 2
    max_batch_per_epoch = None
    train_epochs = 164
    batch_size = 256
    rnn_n_hidden = 200
    rnn_n_layers = 2
    rnn_tie_weights = True
    rnn_clip = 0.25
    rnn_bptt_len = 35
    drop_rate = 0.0
    rnn_weight_norm = False
    dtype = 'fp32'

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    train_set = Wikitext2(dataset_dir, download=True, train=True)
    val_set = Wikitext2(dataset_dir, text_field=train_set.text_field, download=False, train=False)

    train_set.text_field.build_vocab(train_set, vectors=None,
                                     vectors_cache=None)

    train_loader, _ = torchtext.data.BPTTIterator.splits(
        (train_set, val_set),
        batch_size=batch_size * world_size,
        bptt_len=rnn_bptt_len,
        device="cuda:0" if use_cuda else None,
        repeat=False,
        shuffle=True,
    )
    _, val_loader = torchtext.data.BPTTIterator.splits(
        (train_set, val_set),
        batch_size=batch_size,
        bptt_len=rnn_bptt_len,
        device="cuda:0" if use_cuda else None,
        shuffle=False,
    )

    n_tokens, emb_size = len(train_set.text_field.vocab), rnn_n_hidden

    model = RNNLM(
        ntoken=n_tokens,
        ninp=emb_size,
        nhid=rnn_n_hidden,
        nlayers=rnn_n_layers,
        tie_weights=rnn_tie_weights,
        dropout=drop_rate,
        weight_norm=rnn_weight_norm,)

    optimizer = CentralizedSGD(
        world_size=world_size,
        model=model,
        lr=0.2,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=False)

    # Create a learning rate scheduler for an optimizer
    scheduler = MultistepLearningRatesWithWarmup(
        optimizer,
        world_size=world_size,
        milestones=[82, 109],
        gamma=0.1,
        lr=0.1,
        warmup_duration=5,
        warmup_linear_scaling=True,
        warmup_init_lr=None)

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss(reduction="mean")

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [
        Perplexity()
    ]

    if light_target:
        goal = task3_time_to_preplexity_light_goal
    else:
        goal = task3_time_to_preplexity_goal

    tracker = Tracker(metrics, run_id, rank, goal=goal)

    dist.barrier()

    tracker.start()

    num_batches_per_device_train = len(train_loader)

    for epoch in range(0, train_epochs):
        _hidden = (
            model.init_hidden(batch_size)
        )

        # configure local step.
        for batch_idx, batch in enumerate(train_loader):
            model.train()

            tracker.train()
            scheduler.step()

            input = batch.text[
                :,
                rank
                * batch_size : (rank + 1)
                * batch_size,
            ]
            target = batch.target[
                :,
                rank
                * batch_size : (rank + 1)
                * batch_size,
            ]

            # repackage the hidden.
            _hidden = (
                model.repackage_hidden(_hidden)
            )

            # inference and get current performance.
            tracker.batch_start()
            optimizer.zero_grad()

            tracker.record_batch_step('init')

            output, _hidden = model(input, _hidden)

            tracker.record_batch_step('forward')

            loss = loss_function(output.view(-1, n_tokens), target.contiguous().view(-1))

            tracker.record_batch_step('loss')

            loss.backward()

            tracker.record_batch_step('backward')

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            clip_grad_norm_(model.parameters(), rnn_clip)
            optimizer.step()
            tracker.batch_end()

            progress = batch_idx / num_batches_per_device_train

            log_to_api = (batch_idx % LOG_EVERY_N_BATCHES == 0
                          or batch_idx == num_batches_per_device_train)

            for metric in metrics:
                metric_value = metric(loss, output, target).item()

                if tracker:
                    tracker.record_metric(
                        metric,
                        metric_value,
                        output.size()[0],
                        log_to_api=log_to_api)

            status = "Epoch {:5.2f} Batch {:4}: ".format(progress, batch_idx)

            logger.info(status + str(tracker))

            tracker.record_loss(loss, 1, log_to_api=True)

        # finish one epoch training and to decide if we want to val our model.
        tracker.validation()
        tracker.validation_start()

        # each worker finish one epoch training.
        model.eval()

        losses = AverageMeter()
        for metric in metrics:
            metric.reset()

        # Each worker computer their own losses and metrics
        with torch.no_grad():

            _hidden = model.init_hidden(batch_size)

            for batch in val_loader:
                _hidden = model.repackage_hidden(_hidden)
                input, target = batch.text, batch.target
                # Inference
                output, _hidden = model(input, _hidden)

                # Compute loss
                loss = loss_function(output.view(-1, n_tokens), target.contiguous().view(-1))

                # Update loss
                losses.update(loss.item(), input.size(0))

                # Update metrics
                for metric in metrics:
                    metric_value = metric(loss, output, target)
                    metric.update(metric_value, input.size(0))

        # Aggregate metrics and loss for all workers
        metrics_averages = {metric: metric.average().item()
                            for metric in metrics}
        loss_average = global_average(losses.sum, losses.count).item()
        tracker.validation_end()

        for metric, value in metrics_averages.items():
            tracker.record_metric(metric, value, log_to_api=True)

            global_metric_value = global_average(value, 1).item()

            if rank == 0:
                tracker.record_stat(
                    "global_{}".format(metric.name),
                    global_metric_value,
                    log_to_api=True)

        if rank == 0:
            logger.info(
                '{} for rank {}:(best epoch {}, current epoch {}): {:.3f}'.format(
                    tracker.primary_metric.name,
                    tracker.rank,
                    tracker.best_epoch,
                    tracker.current_epoch,
                    tracker.best_metric_value))

        tracker.record_loss(loss, log_to_api=True)

        global_loss = global_average(loss_average, 1).item()

        if rank == 0:
            tracker.record_stat(
                "global_loss",
                global_loss,
                log_to_api=True)
        tracker.validation_end()

        tracker.epoch_end()

        if tracker.goal_reached:
            print("Goal Reached!")
            return
        # train_round(train_loader, model, optimizer, loss_function, metrics,
        #             scheduler, 'fp32', schedule_per='epoch',
        #             transform_target_type=None, use_cuda=use_cuda,
        #             max_batch_per_epoch=max_batch_per_epoch,
        #             tracker=tracker)

        # is_best = validation_round(val_loader, model,  loss_function,
        #                            metrics, run_id, rank, 'fp32',
        #                            transform_target_type=None,
        #                            use_cuda=use_cuda,
        #                            max_batch_per_epoch=max_batch_per_epoch,
        #                            tracker=tracker)

        # checkpointer.save(tracker, model,
        #                   optimizer, scheduler,
        #                   tracker.current_epoch, is_best)

        # tracker.epoch_end()

        # if tracker.goal_reached:
        #     print("Goal Reached!")
        #     return


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
    dataset_dir = os.path.join(args.root_dataset, 'torch', 'wikitext')
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(args.run_id, dataset_dir, ckpt_run_dir,
         output_dir, validation_only=args.validation_only, gpu=args.gpu,
         light_target=args.light)
