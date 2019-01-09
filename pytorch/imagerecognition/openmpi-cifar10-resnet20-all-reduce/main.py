from mlbench_core.dataset.imagerecognition.pytorch import CIFAR10V1, partition_dataset_by_rank
from mlbench_core.models.pytorch.resnet import get_resnet_model
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.lr_scheduler.pytorch import multistep_learning_rates_with_warmup
from mlbench_core.controlflow.pytorch import TrainValidation
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import Checkpointer

from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

import argparse
import os

config = {
    'seed': 42,
    'comm_backend': 'mpi',
    'logging_level': 'DEBUG',
    'logging_file': '/mlbench.log',
    'checkpoint_root': '/checkpoint',
    'train_epochs': 164,
    'batch_size': 256,
    'num_parallel_workers': 2,
    'lr_per_sample': 0.000390625,
    'dataset_root': '/datasets/torch/cifar10',
    'num_classes': 10,
    'momentum': 0.9,
    'nesterov': True,
    'weight_decay': 0.0001,
    'multisteplr_milestones': [82, 109],
    'multisteplr_gamma': 0.1,
    'warmup_linear_scaling': True,
    'warmup_duration': 5,
    'use_cuda': True,
    "dtype": "fp32"
}


def main(run_id):
    checkpoint_dir = os.path.join(config['checkpoint_root'], run_id)
    rank, world_size, _ = initialize_backends(
        comm_backend=config['comm_backend'],
        logging_level=config['logging_level'],
        logging_file=config['logging_file'],
        use_cuda=config['use_cuda'],
        seed=config['seed'],
        ckpt_run_dir=checkpoint_dir
    )

    os.makedirs(config['dataset_root'], exist_ok=True)

    train_set = CIFAR10V1(config['dataset_root'], train=True, download=True)
    val_set = CIFAR10V1(config['dataset_root'], train=False, download=True)

    train_set = partition_dataset_by_rank(train_set, rank, world_size)

    # Set batchsize according to number of workers
    config['batch_size'] = config['batch_size'] // world_size

    train_loader = DataLoader(
        train_set, batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_parallel_workers'],
        pin_memory=config['use_cuda'], drop_last=False)
    val_loader = DataLoader(
        val_set, batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_parallel_workers'],
        pin_memory=config['use_cuda'], drop_last=False)

    model = get_resnet_model('resnet20', 2, 'fp32',
                             num_classes=config['num_classes'], use_cuda=True)

    if config['use_cuda']:
        model.cuda()

    lr = config['lr_per_sample'] * config['batch_size']

    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=config['momentum'],
        weight_decay=config['weight_decay'],
        nesterov=config['nesterov'])

    scheduler = multistep_learning_rates_with_warmup(
        optimizer,
        world_size,
        lr,
        config['multisteplr_gamma'],
        config['multisteplr_milestones'],
        warmup_duration=config['warmup_duration'],
        warmup_linear_scaling=config['warmup_linear_scaling'],
        warmup_lr=lr)

    loss_function = CrossEntropyLoss()

    if config['use_cuda']:
        loss_function.cuda()

    metrics = [TopKAccuracy(topk=1), TopKAccuracy(topk=5)]

    checkpointer = Checkpointer(checkpoint_dir, rank)

    controlflow = TrainValidation(
        model,
        optimizer,
        loss_function,
        metrics,
        scheduler,
        config['batch_size'],
        config['train_epochs'],
        rank,
        world_size,
        run_id,
        dtype=config['dtype'],
        checkpoint=checkpointer,
        use_cuda=config['use_cuda'])

    controlflow.run(dataloader_train=train_loader, dataloader_val=val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, help='The id of the run')
    args = parser.parse_args()
    main(args.run_id)
