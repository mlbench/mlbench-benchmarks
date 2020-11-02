import torch
import torch.distributed as dist
from torch.optim import Adam

from mlbench_core.optim.pytorch.centralized import CustomCentralizedOptimizer
from mlbench_core.optim.pytorch.fp_optimizers import FP16Optimizer
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average


def build_optimizer(model, math, grad_clip, loss_scaling, lr, use_cuda, use_horovod):
    params = model.parameters()

    if math == "fp32":
        optimizer = Adam(params=params, lr=lr)
        fp_optimizer = CustomCentralizedOptimizer(
            optimizer=optimizer,
            model=model,
            grad_clip=grad_clip,
            world_size=dist.get_world_size(),
            use_cuda=use_cuda,
            average_custom=True,
            average_world=False,
        )

    elif math == "fp16":
        model = model.half()  # Set model to half precision
        fp_optimizer = FP16Optimizer(
            model,
            world_size=dist.get_world_size(),
            grad_clip=grad_clip,
            use_cuda=use_cuda,
            use_horovod=use_horovod,
            average_custom=True,
            average_world=False,
            init_scale=loss_scaling["init_scale"],
            scale_window=loss_scaling["upscale_interval"],
            max_scale=8192,
        )

        # Keep params in fp32 for optimizer
        params = [fp_optimizer.fp32_params]
        optimizer = Adam(params=params, lr=lr)
        fp_optimizer.set_optimizer(optimizer)
    else:
        raise NotImplementedError("Unknown math mode {}".format(math))

    return fp_optimizer, optimizer, model


def prepare_batch(data, target, use_cuda=False):
    if use_cuda:
        data = data[0].cuda(), data[1].cuda()
        target = target[0].cuda(), target[1]
    return data, target


def compute_model_output(model, src, trg):
    """Computes output of GNMT model

    Args:
        model (`obj`:torch.nn.Module): The GNMT Model
        src (tuple): Source data point. Should be tuple of (tokens, lengths)
        trg (tuple): Target data point. Should be tuple of (tokens, lengths)
    Returns:
        `obj`:torch.Tensor: The output tensor
    """
    return model(src[0], src[1], trg[0][:-1])


def compute_loss(src, trg, output, loss_func, iter_size):
    """Computes the Loss of a given input and output

    Args:
        src (tuple): Source data point. Should be tuple of (tokens, lengths)
        trg (tuple): Target data point. Should be tuple of (tokens, lengths)
        output (`obj`:torch.Tensor): Output of given input
        loss_func (`obj`:torch.nn.Module): Loss function
        iter_size (int): Number of iterations to do before calling `optimizer.step`
    Returns:
        (`obj`:torch.Tensor, float): Total loss, loss per token
    """
    src, src_length = src
    trg, trg_length = trg

    num_toks = {"trg": int(sum(trg_length - 1)), "src": int(sum(src_length))}

    tgt_labels = trg[1:]
    T, B = output.size(0), output.size(1)

    loss = loss_func(output.view(T * B, -1), tgt_labels.contiguous().view(-1))

    loss_per_batch = loss.item()
    loss /= B * iter_size

    loss_per_token = loss_per_batch / num_toks["trg"]

    return loss, loss_per_token


def validation_round(
    val_loader,
    metrics,
    model,
    loss_func,
    iter_size,
    translator,
    tracker=None,
    use_cuda=False,
):
    # Set tracker and model in eval mode
    model.eval()
    if tracker:
        tracker.validation()
        tracker.validation_start()

    losses = AverageMeter()

    # Reset metrics
    for metric in metrics:
        metric.reset()

    with torch.no_grad():
        for (data, target) in val_loader:
            data, target = prepare_batch(data, target, use_cuda=use_cuda)
            output = compute_model_output(model, data, target)

            # Compute loss
            loss, loss_per_token = compute_loss(
                data, target, output, loss_func, iter_size
            )

            # Update loss
            losses.update(loss_per_token, 1)

            # Update metrics
            translated, targets = translator.translate(data, target)
            for metric in metrics:
                metric_value = metric(loss, translated, targets)
                size = data[0].shape[1]

                metric.update(metric_value, size)

    metrics_averages = {metric: metric.average().item() for metric in metrics}
    loss_average = global_average(losses.sum, losses.count).item()

    if tracker:
        tracker.validation_end()
    return metrics_averages, loss_average
