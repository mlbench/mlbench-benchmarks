# -*- coding: utf-8 -*-
from copy import deepcopy

import numpy as np
import torch

from pcode.utils.checkpoint import save_to_checkpoint
from pcode.utils.logging import (
    display_training_stat,
    display_test_stat,
    dispaly_best_test_stat,
)
from pcode.utils.stat_tracker import RuntimeTracker
from pcode.utils.timer import Timer
from pcode.utils.auxiliary import get_model_difference
import pcode.utils.error_handler as error_handler
from pcode.create_dataset import load_data_batch

from mlbench_core.utils import Tracker
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy


# sys.excepthook = error_handler.global_except_hook


def train_and_validate(
    conf, model, criterion, scheduler, optimizer, metrics, data_loader
):
    print("=>>>> start training and validation.\n")

    # define runtime stat tracker and start the training.
    tracker_tr = RuntimeTracker(metrics_to_track=metrics.metric_names)
    mlbench_metrics = [
        TopKAccuracy(topk=1)
    ]
    mlbench_tracker = Tracker(mlbench_metrics, conf.run_id, conf.rank)

    # define the timer for different operations.
    timer = Timer(
        verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
        log_fn=conf.logger.log_metric,
    )

    # break until finish expected full epoch training.
    print("=>>>> enter the training.\n")
    mlbench_tracker.start()
    while True:
        # init the hidden state.
        _hidden = (
            model.module.init_hidden(conf.batch_size)
            if "DataParallel" == model.__class__.__name__
            else model.init_hidden(conf.batch_size)
        )

        # configure local step.
        for batch in data_loader["train_loader"]:
            model.train()

            mlbench_tracker.train()
            scheduler.step(optimizer)

            # repackage the hidden.
            _hidden = (
                model.module.repackage_hidden(_hidden)
                if "DataParallel" == model.__class__.__name__
                else model.repackage_hidden(_hidden)
            )

            # load data
            with timer("load_data", epoch=scheduler.epoch_):
                _input = batch.text[
                    :,
                    conf.graph.rank
                    * conf.batch_size : (conf.graph.rank + 1)
                    * conf.batch_size,
                ]
                _target = batch.target[
                    :,
                    conf.graph.rank
                    * conf.batch_size : (conf.graph.rank + 1)
                    * conf.batch_size,
                ]
                _input, _target = load_data_batch(conf, _input, _target)

            # inference and get current performance.
            with timer("forward_pass", epoch=scheduler.epoch_):
                mlbench_tracker.batch_start()
                optimizer.zero_grad()
                loss, _hidden = inference(
                    conf,
                    model,
                    criterion,
                    metrics,
                    _input,
                    _target,
                    _hidden,
                    tracker_tr,
                    mlbench_tracker
                )
                mlbench_tracker.record_batch_step('loss')

            with timer("backward_pass", epoch=scheduler.epoch_):
                loss.backward()
                mlbench_tracker.record_batch_step('backward')

            with timer("sync_complete", epoch=scheduler.epoch_):
                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.rnn_clip)
                n_bits_to_transmit = optimizer.step(timer=timer)
                mlbench_tracker.batch_end()

            mlbench_tracker.record_loss(loss, 1, log_to_api=True)

            # display the logging info.
            display_training_stat(conf, scheduler, tracker_tr, n_bits_to_transmit)

            # finish one epoch training and to decide if we want to val our model.
            if scheduler.epoch_ % 1 == 0:
                if tracker_tr.stat["loss"].avg > 1e3 or np.isnan(
                    tracker_tr.stat["loss"].avg
                ):
                    print("\nThe process diverges!!!!!Early stop it.")
                    error_handler.abort()

                mlbench_tracker.validation()
                mlbench_tracker.validation_start()

                # each worker finish one epoch training.
                do_validate(
                    conf, model, optimizer, criterion, scheduler, metrics,
                    data_loader, mlbench_tracker
                )
                mlbench_tracker.validation_end()

                # refresh the logging cache at the begining of each epoch.
                tracker_tr.reset()

                # determine if the training is finished.
                if scheduler.is_stop():
                    conf.logger.save_json()
                    return

            mlbench_tracker.epoch_end()

            # display tracking time.
            if (
                conf.graph.rank == 0
                and conf.display_tracked_time
                and scheduler.local_index % conf.summary_freq == 0
            ):
                print(timer.summary())


def inference(conf, model, criterion, metrics, _input, _target, _hidden, tracker=None, mlbench_tracker=None):
    """Inference on the given model and get loss and accuracy."""
    output, _hidden = model(_input, _hidden)
    if mlbench_tracker:
        mlbench_tracker.record_batch_step('forward')
    loss = criterion(output.view(-1, conf.n_tokens), _target.contiguous().view(-1))
    performance = metrics.evaluate(loss, output, _target)
    if tracker is not None:
        tracker.update_metrics([loss.item()] + performance, n_samples=_input.size(0))
    return loss, _hidden


def do_validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader, mlbench_tracker):
    """Evaluate the model on the test dataset and save to the checkpoint."""
    # wait until the whole group enters this function, and then evaluate.
    performance = validate(
        conf, model, optimizer, criterion, scheduler, metrics, data_loader
    )

    mlbench_tracker.record_stat("ppl", performance[0], log_to_api=True)

    # remember best performance and display the val info.
    scheduler.best_tracker.update(performance[0], scheduler.epoch_)
    dispaly_best_test_stat(conf, scheduler)

    # save to the checkpoint.
    save_to_checkpoint(
        conf,
        {
            "arch": conf.arch,
            "current_epoch": scheduler.epoch,
            "local_index": scheduler.local_index,
            "best_perf": scheduler.best_tracker.best_perf,
            "optimizer": optimizer.state_dict(),
            "state_dict": model.state_dict(),
        },
        scheduler.best_tracker.is_best,
        dirname=conf.checkpoint_dir,
        filename="checkpoint.pth.tar",
        save_all=conf.save_all_models,
    )
    print("Finished validation.")


def validate(conf, model, optimizer, criterion, scheduler, metrics, data_loader):
    """A function for model evaluation."""

    def _evaluate(_model, label):
        # define stat.
        tracker_te = RuntimeTracker(metrics_to_track=metrics.metric_names)

        # switch to evaluation mode
        _model.eval()

        # define hidden state for RNN.
        _hidden = (
            model.module.init_hidden(conf.batch_size)
            if "DataParallel" == model.__class__.__name__
            else model.init_hidden(conf.batch_size)
        )

        for batch in data_loader["val_loader"]:
            # load data and check performance.
            _input, _target = batch.text, batch.target

            # repackage the hidden.
            _hidden = (
                model.module.repackage_hidden(_hidden)
                if "DataParallel" == model.__class__.__name__
                else model.repackage_hidden(_hidden)
            )

            with torch.no_grad():
                _, _hidden = inference(
                    conf,
                    _model,
                    criterion,
                    metrics,
                    _input,
                    _target,
                    _hidden,
                    tracker_te,
                )

        # display the test stat.
        display_test_stat(conf, scheduler, tracker_te, label)

        # get global (mean) performance
        global_performance = tracker_te.evaluate_global_metrics()
        return global_performance

    # # evaluate the averaged local model on the validation dataset.
    # if (
    #     conf.graph_topology != "complete"
    #     and conf.graph_topology != "data_center"
    #     and not conf.train_fast
    # ):
    #     copied_model = deepcopy(model)
    #     optimizer.world_aggregator.agg_model(copied_model, op="avg")
    #     _evaluate(copied_model, label="averaged_model")

    #     # get the l2 distance of the local model to the averaged model
    #     conf.logger.log_metric(
    #         name="stat",
    #         values={
    #             "rank": conf.graph.rank,
    #             "epoch": scheduler.epoch_,
    #             "distance": get_model_difference(model, copied_model),
    #         },
    #         tags={"split": "test", "type": "averaged_model"},
    #     )

    # evaluate each local model on the validation dataset.
    global_performance = _evaluate(model, label="local_model")
    return global_performance
