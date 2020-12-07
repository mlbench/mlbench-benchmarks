import numpy as np
import torch
from torch import distributed as dist
from torch.optim import Adam

from mlbench_core.optim.pytorch.centralized import CustomCentralizedOptimizer
from mlbench_core.optim.pytorch.fp_optimizers import FP16Optimizer
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average


def build_optimizer(
    model,
    optimizer_args,
    math_mode="fp16",
    scaling_args=None,
    use_horovod=False,
    use_cuda=False,
):
    """Builds the Floating Point optimizer for Transformer. Uses Adam or FusedAdam as
    underlying optimizer

    Args:
        model (:obj:`nn.Module`): The model
        optimizer_args (dict): The arguments for optimizer (eps, weight_decay)
        math_mode (str): One of `fp32` or `fp16`. Default `fp16`
        scaling_args (dict): Arguments for loss scaling (for `float16`). Default `None`
        use_horovod (bool): Use horovod for communication
        use_cuda (bool): Use CUDA tensors for communication

    Returns:
        (:obj:`FP16Optimizer` | :obj:`FP32Optimizer`, :obj:`torch.optim.Optimizer`, :obj:`nn.Module`):
            The FPoptimizer, its underlying Adam or FusedAdam and the model
    """
    if math_mode == "fp16":
        # Half model
        model.half()

        # Build fp16_optimizer
        fp_optimizer = FP16Optimizer(
            fp16_model=model,
            world_size=dist.get_world_size(),
            use_cuda=use_cuda,
            use_horovod=use_horovod,
            average_custom=True,
            divide_before=True,
            **scaling_args
        )
        params = [fp_optimizer.fp32_params]
        optimizer = Adam(params=params, **optimizer_args)
        fp_optimizer.set_optimizer(optimizer)

    elif math_mode == "fp32":
        optimizer = Adam(params=model.parameters(), **optimizer_args)
        fp_optimizer = CustomCentralizedOptimizer(
            optimizer=optimizer,
            model=model,
            world_size=dist.get_world_size(),
            use_cuda=use_cuda,
            average_custom=True,
        )

    else:
        raise NotImplementedError("Unknown math mode {}".format(math_mode))

    return fp_optimizer, optimizer, model


def opt_step(
    fp_optimizer, tracker, full_batch_size, update_freq, math_mode, world_size
):
    """Performs one optimizer step.

    Args:
        fp_optimizer (:obj:`FP16Optimizer` | :obj:`FP32Optimizer`): The FP Optimizer
        tracker (:obj:`mlbench_core.utils.Tracker`, optional) The current tracker
        full_batch_size (int): The total batch size (over all batches since last update)
        update_freq (int): The update frequency between batches
        math_mode (str): The used math mode
        world_size (int): Distributed world size

    Returns:
        (bool): Whether the weights were updated or not (i.e. if no overflow detected)
    """
    if math_mode == "fp32":
        updated = fp_optimizer.step(tracker=tracker, denom=full_batch_size)
    elif math_mode == "fp16":
        # This results in reducing tensor and dividing by `loss_scale * full_batch_size`
        # but we divide by world size before reduction to avoid overflow, and
        # re-multiply after reduction to rescale
        multiplier = full_batch_size / (world_size * update_freq)
        denom = world_size * update_freq
        updated = fp_optimizer.step(tracker=tracker, denom=denom, multiplier=multiplier)
    else:
        raise NotImplementedError("Unknown math mode {}".format(math_mode))
    return updated


def compute_loss(batch, output, loss_func):
    """Computes the loss of a given batch

    Args:
        batch (dict): The current batch
        output (tuple): Tuple of tensors (output of `TransformerModel`)
        loss_func (:obj:`nn.modules._Loss`): The loss function

    Returns:
        (:obj:`torch.Tensor`, int): The computed loss and the total number of tokens in batch
    """
    output = output[0]
    T, B = output.size(0), output.size(1)
    target = batch["target"]

    loss = loss_func(output.view(T * B, -1), target.contiguous().view(-1))
    return loss, batch["ntokens"]


def get_full_batch_size(rank_ntokens, world_size=1, use_cuda=False):
    """Returns the full batch size over all workers

    Args:
        rank_ntokens (int): Number of tokens in current worker
        world_size (int): Distributed world size
        use_cuda (bool): Use CUDA tensors for communication

    Returns:
        (int): The sum of all workers' batch size for current batch
    """
    tensor = torch.tensor(
        [rank_ntokens], device=torch.device("cuda" if use_cuda else "cpu")
    )
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor.item()


def validation_round(loader, metrics, criterion, translator, tracker, use_cuda=False):
    """Performs one round of validation for the Transformer model

    Args:
        loader (:obj:`torch.utils.data.DataLoader`): Data loader
        metrics (list): List of metrics for evaluation
        criterion (:obj:`torch.nn.Module): Loss function
        translator (:obj:`mlbench_core.models.pytorch.transformer.SequenceGenerator`): Translator module
        tracker (:obj:`mlbench_core.utils.Tracker`): Current Tracker
        use_cuda (bool): Use GPU acceleration. Default: `False`.

    Returns:
        (dict of :obj:`mlbench_core.evaluation.pytorch.MLBenchMetric`: float, float):
            The metrics averages over all workers, and the loss average.
    """
    model = translator.model
    model.eval()
    tracker.validation()
    tracker.validation_start()

    losses = AverageMeter()
    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        for batch in loader:
            batch = prepare_batch(batch, use_cuda=use_cuda)
            output = model(**batch["net_input"])

            loss, sample_size = compute_loss(batch, output, criterion)

            losses.update(loss.item() / sample_size, 1)

            translated, targets = translator.translate_batch(batch)
            for metric in metrics:
                metric_value = metric(translated, targets)
                size = batch["target"].size(0)  # Number of translated sentences
                metric.update(metric_value, size)

    metric_averages = {metric: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count)

    tracker.validation_end()

    return metric_averages, loss_average


class Arguments:
    def __init__(self, args):
        self.args = args

    def __getattr__(self, name):
        if name in self.args:
            return self.args[name]
        else:
            raise AttributeError


def prepare_batch(sample, use_cuda=False):
    """Prepares a batch for training/validation by moving it to GPU

    Args:
        sample (dict | :obj:`torch.Tensor` | list): The batch to prepare
        use_cuda (bool): Move to cuda

    Returns:
        (dict | :obj:`torch.Tensor` | list): The batch, on GPU or CPU
    """

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    if use_cuda:
        return _move_to_cuda(sample)
    else:
        return sample


def equalize_batches(batches, world_size, seed):
    """Given a list of batches, makes sure each workers has equal number
    by adding new batches using bootstrap sampling

    Args:
        batches (list): The list of batches
        world_size (int): Distributed world size
        seed (int): Random seed to use (must be the same across all workers)

    Returns:
        (list): The new extended batches
    """
    to_add = world_size - (len(batches) % world_size)
    if to_add == 0:
        return batches
    np.random.seed(seed)
    bootstrapped = np.random.choice(np.arange(len(batches)), size=to_add)

    to_add = [batches[i] for i in bootstrapped]
    return batches + to_add
