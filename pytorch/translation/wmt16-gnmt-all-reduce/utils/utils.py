import logging

import torch
from apex import amp
from torch.optim import Adam

from mlbench_core.optim.pytorch.fp_optimizers import (
    AMPOptimizer,
    FP16Optimizer,
    FP32Optimizer,
)
from mlbench_core.utils import AverageMeter
from mlbench_core.utils.pytorch.distributed import global_average

logger = logging.getLogger("mlbench")
LOG_EVERY_N_BATCHES = 25


def build_optimizer(
    model, math, grad_clip, loss_scaling, lr, use_cuda, world_size, use_horovod
):
    params = model.parameters()
    if math == "amp_fp16":
        optimizer = Adam(params=params, lr=lr)
        model, optimizer = amp.initialize(
            model,
            optimizer,
            cast_model_outputs=torch.float16,
            keep_batchnorm_fp32=False,
            opt_level="O2",
        )

        fp_optimizer = AMPOptimizer(
            model,
            grad_clip=grad_clip,
            loss_scale=loss_scaling["init_scale"],
            dls_upscale_interval=loss_scaling["upscale_interval"],
            use_cuda=use_cuda,
            world_size=world_size,
            average_custom=True,
            average_world=False,
            use_horovod=use_horovod,
        )
    else:

        if math == "fp32":
            fp_optimizer = FP32Optimizer(
                model=model,
                grad_clip=grad_clip,
                world_size=world_size,
                use_cuda=use_cuda,
            )

        elif math == "fp16":
            model = model.half()  # Set model to half precision
            fp_optimizer = FP16Optimizer(
                model,
                world_size=world_size,
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
        else:
            return NotImplementedError()

        optimizer = Adam(params=params, lr=lr)
    fp_optimizer.set_optimizer(optimizer)
    return fp_optimizer, optimizer, model


def prepare_batch(data, target, use_cuda=False):
    if use_cuda:
        data = data[0].cuda(), data[1].cuda()
        target = target[0].cuda(), target[1]
    return data, target


def compute_model_output(model, src, trg):
    """ Computes output of GNMT model

    Args:
        model (`obj`:torch.nn.Module): The GNMT Model
        src (tuple): Source data point. Should be tuple of (tokens, lengths)
        trg (tuple): Target data point. Should be tuple of (tokens, lengths)
    Returns:
        `obj`:torch.Tensor: The output tensor
    """
    return model(src[0], src[1], trg[0][:-1])


def compute_loss(src, trg, output, loss_func, iter_size):
    """ Computes the Loss of a given input and output

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


def opt_step(fp_optimizer, update_freq, math_mode, world_size, tracker):
    """Performs one optimizer step.
    Args:
        fp_optimizer (:obj:`FP16Optimizer` | :obj:`FP32Optimizer`): The FP Optimizer
        update_freq (int): The update frequency between batches
        math_mode (str): The used math mode
        world_size (int): Distributed world size
        tracker: (:obj:`mlbench_core.utils.Tracker`): Current tracker object
    Returns:
        (bool): Whether the weights were updated or not (i.e. if no overflow detected)
    """
    if math_mode == "fp32":
        updated = fp_optimizer.step(tracker=tracker)
    elif math_mode == "fp16" or math_mode == "amp_fp16":
        # Divide gradients by world_size*update_freq
        denom = world_size * update_freq
        updated = fp_optimizer.step(tracker=tracker, denom=denom)
    else:
        raise NotImplementedError

    return updated


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
