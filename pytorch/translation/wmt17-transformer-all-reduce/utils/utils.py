import torch
from torch import distributed as dist
from torch.optim import Adam

from mlbench_core.optim.pytorch.fp_optimizers import FP16Optimizer, FP32Optimizer
# from apex.optimizers.fused_adam import FusedAdam


def build_optimizer(model, optimizer_args, scaling_args, math_mode, use_horovod=False, use_cuda=False):
    if math_mode == "fp16":
        # Half model
        model = model.half()

        # Build fp16_optimizer
        fp_optimizer = FP16Optimizer(fp16_model=model,
                                     world_size=dist.get_world_size(),
                                     use_cuda=use_cuda,
                                     use_horovod=use_horovod,
                                     average_models=True,
                                     divide_before=True,
                                     **scaling_args)
        params = [fp_optimizer.fp32_params]

    elif math_mode == "fp32":
        fp_optimizer = FP32Optimizer(model=model,
                                     world_size=dist.get_world_size(),
                                     use_cuda=use_cuda,
                                     average_batch=True)
        params = model.parameters()

    else:
        raise NotImplementedError("Unknown math mode {}".format(math_mode))

    optimizer = Adam(params=params, **optimizer_args)
    # optimizer = FusedAdam(params=params, **optimizer_args)
    fp_optimizer.set_optimizer(optimizer)

    return fp_optimizer, optimizer, model


def opt_step(fp_optimizer, full_batch_size, math_mode, world_size):

    if math_mode == "fp32":
        updated = fp_optimizer.step(denom=full_batch_size)
    elif math_mode == "fp16":
        # This results in reducing tensor and dividing by `loss_scale * full_batch_size`
        # but we divide by world size before reduction to avoid overflow, and
        # re-multiply after reduction to rescale
        multiplier = full_batch_size / world_size
        updated = fp_optimizer.step(multiplier=multiplier)
    else:
        raise NotImplementedError("Unknown math mode {}".format(math_mode))
    return updated


def compute_loss(batch, output, loss_func):
    output = output[0]
    T, B = output.size(0), output.size(1)
    target = batch['target']

    loss = loss_func(output.view(T * B, -1), target.contiguous().view(-1))
    return loss, batch['ntokens']


def get_full_batch_size(rank_ntokens, world_size=1, use_cuda=False):
    tensor = torch.tensor([rank_ntokens], device=torch.device("cuda" if use_cuda else "cpu"))
    if world_size > 1:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor.item()


class Arguments:
    def __init__(self, args):
        self.args = args

    def __getattr__(self, name):
        if name in self.args:
            return self.args[name]
        else:
            raise AttributeError


def prepare_batch(sample, use_cuda=False):
    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda()
        elif isinstance(maybe_tensor, dict):
            return {
                key: _move_to_cuda(value)
                for key, value in maybe_tensor.items()
            }
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    if use_cuda:
        return _move_to_cuda(sample)
    else:
        return sample
