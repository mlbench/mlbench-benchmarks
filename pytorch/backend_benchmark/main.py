import argparse
import logging
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from mlbench_core.utils import Tracker
from mlbench_core.utils.pytorch import initialize_backends

try:
    import horovod.torch as hvd
except ImportError as e:
    hvd = None
    pass
logger = logging.getLogger("mlbench")


def get_random_tensor(size, dtype, use_cuda):
    """Returns a random tensor of given type and size

    Args:
        size (int): Tensor size (number of elements)
        dtype (:obj:`torch.dtype`): One of `torch.float16` and `torch.flaot32`
        use_cuda (bool): Return CUDA tensor

    Returns:

    """
    tensor = torch.rand(size).to(dtype=dtype)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def reduce_tensor(tensor, use_horovod):
    """Reduces the given tensor across all workers

    Args:
        tensor (:obj:`torch.Tensor`): THe tensor to reduce
        use_horovod (bool): Use horovod for communication
    """
    if use_horovod:
        tensor = hvd.allreduce(tensor, op=hvd.Sum)
    else:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)


def get_communication_average(size, dtype, use_cuda, num_samples, use_horovod):
    """Performs multiple reductions of random tensors, and returns the average time
    per operation.

    Args:
        size (int): Tensor size
        dtype (:obj:`torch.dtype`): One of `torch.float16` and `torch.flaot32`
        use_cuda (bool): Use CUDA tensors
        num_samples (int): Number of samples to gather
        use_horovod (bool): Use horovod for communication

    Returns:
        (float): Average time per communication step in seconds
    """
    times = []
    for i in range(num_samples):
        tensor = get_random_tensor(size, dtype=dtype, use_cuda=use_cuda)
        start = datetime.now()
        reduce_tensor(tensor, use_horovod=use_horovod)
        if use_cuda:
            torch.cuda.synchronize()
        end = datetime.now()
        times.append((end - start).total_seconds())

    return sum(times) / len(times)


def verify_communication(use_horovod, world_size):
    """Verifies that the communication between workers works as expected
    It reduces a tensor of [1], and verifies that the reduced tensor is the same as the world size
    Args:
        use_horovod (bool): Use horovod for communication
        world_size (int): Distributed world size

    Raises:
        AssertionError: if the communication doesn't work as expected
    """
    if use_horovod:
        hvd.init()
        logger.info("Using horovod, rank = {}".format(hvd.rank()))
        tensor = torch.tensor(
            [1],
            device=torch.device(
                "cuda" if dist.get_backend() == dist.Backend.NCCL else "cpu"
            ),
        )
        res = hvd.allreduce(tensor, op=hvd.Sum)
        assert res[0] == world_size, "Communication is not working"
    else:
        logger.info("Using torch, rank={}".format(dist.get_rank()))
        tensor = torch.tensor(
            [1],
            device=torch.device(
                "cuda" if dist.get_backend() == dist.Backend.NCCL else "cpu"
            ),
        )
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        assert tensor[0] == world_size, "Communication is not working"
    if hvd:
        logger.info(
            "NCCL Built={}, MPI Built={} , GLOO Built={}".format(
                hvd.nccl_built(), hvd.mpi_built(), hvd.gloo_built()
            )
        )


def train_loop(run_id, use_horovod=False, gpu=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    # Define size range and number of samples to gather
    size_range = np.logspace(0, 8, num=80)
    num_samples = 100

    do_fp16 = True#dist.get_backend() != dist.Backend.MPI or use_horovod
    is_nccl = dist.get_backend() == dist.Backend.NCCL

    logger.info("Using {}".format(dist.get_backend()))
    # Verify that communication works
    verify_communication(use_horovod, world_size)

    tracker = Tracker([], run_id, rank)
    dist.barrier()
    tracker.start()
    tracker.validation()
    tracker.validation_start()

    # Perform benchmark on both GPU and CPU (except for NCCL)
    if is_nccl and not gpu:
        raise ValueError("Cannot run NCCL without GPU")

    for j, size in enumerate(size_range):
        size = int(size)
        avg = get_communication_average(
            size, torch.float32, gpu, num_samples, use_horovod
        )
        tracker.record_stat("tensor_size", size, log_to_api=True)
        tracker.record_stat("dtype", 32, log_to_api=True)
        tracker.record_stat("cuda", 1 if gpu else 0, log_to_api=True)
        tracker.record_stat("avg_time", avg, log_to_api=True)
        logger.info(
            "Size={}, dtype=float32, use_cuda={}, avg_time={}".format(
                size, gpu, avg
            )
        )

        if do_fp16:
            avg = get_communication_average(
                size, torch.float16, gpu, num_samples, use_horovod
            )
            tracker.record_stat("tensor_size", size, log_to_api=True)
            tracker.record_stat("dtype", 16, log_to_api=True)
            tracker.record_stat("cuda", 1 if gpu else 0, log_to_api=True)
            tracker.record_stat("avg_time", avg, log_to_api=True)
            logger.info(
                "Size={}, dtype=float16, use_cuda={}, avg_time={}".format(
                    size, gpu, avg
                )
            )
    tracker.validation_end()
    time.sleep(10)


def main(run_id, output_dir, rank, backend, hosts, use_horovod=False, gpu=False):
    r"""Main logic."""
    with initialize_backends(
        comm_backend=backend,
        hosts=hosts,
        rank=rank,
        logging_level="INFO",
        logging_file=os.path.join(output_dir, "mlbench.log"),
        use_cuda=gpu,
        seed=42,
        cudnn_deterministic=False,
        delete_existing_ckpts=True,
    ):
        train_loop(run_id=run_id, use_horovod=use_horovod, gpu=gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process run parameters")
    parser.add_argument("--run_id", type=str, default="1", help="The id of the run")
    parser.add_argument(
        "--root-output",
        type=str,
        default="/output",
        help="Default root directory to output.",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False, help="Train with GPU"
    )
    parser.add_argument(
        "--horovod",
        action="store_true",
        default=False,
        help="Use horovod for communication (should be installed)",
    )
    parser.add_argument("--rank", type=int, default=1, help="The rank of the process")
    parser.add_argument(
        "--backend", type=str, default="mpi", help="PyTorch distributed backend"
    )
    parser.add_argument("--hosts", type=str, help="The list of hosts")

    args = parser.parse_args()

    uid = "benchmark"

    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(output_dir)
    main(
        args.run_id,
        output_dir,
        rank=args.rank,
        backend=args.backend,
        hosts=args.hosts,
        use_horovod=args.horovod,
        gpu=args.gpu,
    )
