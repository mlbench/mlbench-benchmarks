#!/usr/bin/env python

import os
import sys
import argparse
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torchvision import datasets, transforms

from mlbench_core.utils import Tracker
from mlbench_core.evaluation.goals import task1_time_to_accuracy_goal
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.controlflow.pytorch import validation_round

import logging


class Partition(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ Partitions a dataset into different chuncks. """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def partition_dataset_train():
    """ Partitioning MNIST train set"""
    dataset = datasets.MNIST(
        './data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return train_set, bsz


def partition_dataset_val():
    """ Partitioning MNIST validation set"""
    dataset = datasets.MNIST(
        './data',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))
        ]))
    size = dist.get_world_size()
    bsz = int(128 / float(size))
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    val_set = torch.utils.data.DataLoader(
        partition, batch_size=bsz, shuffle=True)
    return val_set, bsz


def average_gradients(model):
    """ Gradient averaging. """
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def run(rank, size, run_id):
    """ Distributed Synchronous SGD Example """
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset_train()
    val_set, bsz_val = partition_dataset_val()
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    metrics = [
        TopKAccuracy(topk=1),
        TopKAccuracy(topk=5)
    ]
    loss_func = nn.NLLLoss()

    goal = task1_time_to_accuracy_goal

    tracker = Tracker(metrics, run_id, rank, goal=goal)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    num_batches_val = ceil(len(val_set.dataset) / float(bsz_val))

    tracker.start()

    for epoch in range(10):
        tracker.train()

        epoch_loss = 0.0
        for data, target in train_set:
            tracker.batch_start()

            optimizer.zero_grad()
            output = model(data)

            tracker.record_batch_step('forward')

            loss = loss_func(output, target)
            epoch_loss += loss.data.item()

            tracker.record_batch_step('loss')

            loss.backward()

            tracker.record_batch_step('backward')

            average_gradients(model)
            optimizer.step()

            tracker.batch_end()

        logging.debug('Rank %s, epoch %s: %s',
                      dist.get_rank(), epoch,
                      epoch_loss / num_batches)

        validation_round(val_set, model, loss_func, metrics, run_id, rank,
                         'fp32', transform_target_type=None, use_cuda=False,
                         max_batch_per_epoch=num_batches_val, tracker=tracker)

        tracker.epoch_end()

        if tracker.goal_reached:
            logging.debug("Goal Reached!")
            return


def init_processes(rank, run_id, hosts, backend='gloo'):
    """ Initialize the distributed environment. """
    hosts = hosts.split(',')
    os.environ['MASTER_ADDR'] = hosts[0]  # first worker is the master worker
    os.environ['MASTER_PORT'] = '29500'
    world_size = len(hosts)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    run(rank, world_size, run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, help='The id of the run')
    parser.add_argument('--rank', type=int, help='The rank of this worker')
    parser.add_argument('--hosts', type=str, help='The list of hosts')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logging.info("Started Training with {}".format(str(args)))

    init_processes(args.rank, args.run_id, args.hosts)
