import torch
from torch.optim import SGD

from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average


def validation_round(
    loader, model, batch_size, n_tokens, metrics, loss_function, tracker
):
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

        hidden = model.init_hidden(batch_size)

        for data, target in loader:
            hidden = model.repackage_hidden(hidden)

            # Inference
            output, hidden = model(data, hidden)

            # Compute loss
            loss = loss_function(
                output.view(-1, n_tokens), target.contiguous().view(-1)
            )

            # Update loss
            losses.update(loss.item(), data.size(0))

            # Update metrics
            for metric in metrics:
                metric_value = metric(output, target)
                metric.update(metric_value, data.size(0))

    # Aggregate metrics and loss for all workers
    metrics_averages = {metric: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()
    tracker.validation_end()

    return metrics_averages, loss_average
