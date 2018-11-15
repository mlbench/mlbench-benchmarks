r"""Example of using mlbench : CIFAR10 + Resnet20 + MPI"""

import argparse
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss

from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.models.pytorch.resnet import ResNetCIFAR
from mlbench_core.lr_scheduler.pytorch.lr import MultiStepLR
from mlbench_core.controlflow.pytorch import TrainValidation
from mlbench_core.utils.pytorch.checkpoint import Checkpointer
from mlbench_core.dataset.imagerecognition.pytorch import CIFAR10V1, partition_dataset_by_rank


def main(run_id):
    r"""Main logic."""
    num_parallel_workers = 2
    dataset_root = '/datasets/torch/cifar10'
    use_cuda = True
    batch_size = 128

    initialize_backends(comm_backend='mpi',
                        logging_level='INFO',
                        logging_file='/mlbench.log',
                        use_cuda=use_cuda,
                        seed=42,
                        cudnn_deterministic=False,
                        ckpt_run_dir='/checkpoints',
                        resume=False)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = ResNetCIFAR(resnet_size=20,
                        bottleneck=False,
                        num_classes=10,
                        version=1)

    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=1e-4,
                          nesterov=True)

    # Create a learning rate scheduler for an optimizer
    scheduler = MultiStepLR(optimizer,
                            milestones=[82, 109],
                            gamma=0.1)

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [TopKAccuracy(topk=1), TopKAccuracy(topk=5)]

    train_set = CIFAR10V1(dataset_root, train=True, download=True)
    val_set = CIFAR10V1(dataset_root, train=False, download=True)

    train_set = partition_dataset_by_rank(train_set, rank, world_size)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda, drop_last=False)

    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda, drop_last=False)

    checkpointer = Checkpointer(
        ckpt_run_dir='/checkpoints',
        rank=rank,
        checkpoint_all=True)

    controlflow = TrainValidation(
        model=model,
        optimizer=optimizer,
        loss_function=loss_function,
        metrics=metrics,
        scheduler=scheduler,
        batch_size=batch_size,
        train_epochs=164,
        rank=rank,
        world_size=world_size,
        run_id=run_id,
        dtype='fp32',
        validate=True,
        schedule_per='epoch',
        checkpoint=checkpointer,
        transform_target_type=None,
        average_models=True,
        use_cuda=True,
        max_batch_per_epoch=None)

    controlflow.run(
        dataloader_train=train_loader,
        dataloader_val=val_loader,
        dataloader_train_fn=None,
        dataloader_val_fn=None,
        resume=False,
        repartition_per_epoch=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, help='The id of the run')
    args = parser.parse_args()
    main(args.run_id)
