import torch
import torch.distributed as dist

from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import get_backend_tensor, global_average


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def set_sequence_lengths(dataset, random=False):
    """Sets the sequences lengths and broadcasts to other workers

    Args:
        dataset (:obj:`mlbench_core.dataset.nlp.pytorch.Wikitext2Dataset`)

    """
    dataset.generate_sequence_lengths(random=random)
    seq_lens = get_backend_tensor(dataset.sequence_lengths)
    dist.broadcast(seq_lens, src=0)
    dataset.sequence_lengths = seq_lens.cpu()


def validation_round(
    val_set, model, batch_size, metrics, loss_function, tracker, use_cuda=False
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

        num_batches = val_set.num_batches()
        for batch_idx in range(num_batches):
            data, target = val_set.get_batch(batch_idx, cuda=use_cuda)
            # Inference
            output, hidden = model(data, hidden)

            # Compute loss
            loss = loss_function(output, target)

            # Update loss
            losses.update(loss.item(), data.size(0))

            hidden = repackage_hidden(hidden)

            # Update metrics
            for metric in metrics:
                metric_value = metric(output, target)
                metric.update(metric_value, data.size(0))

    # Aggregate metrics and loss for all workers
    metrics_averages = {metric: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()
    tracker.validation_end()

    return metrics_averages, loss_average
