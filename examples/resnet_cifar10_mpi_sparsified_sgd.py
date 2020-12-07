r"""Example of using sparsified sgd with memory : CIFAR10 + Resnet20 + MPI + GPU

.. code-block:: bash
    mpirun -n 2 --oversubscribe python resnet_cifar10_mpi_sparsified_sgd.py --run_id 1
"""
import argparse
import json
import os

from mlbench_core.controlflow.pytorch.train_validation import TrainValidation
from mlbench_core.controlflow.pytorch.checkpoints_evaluation import CheckpointsEvaluationControlFlow
from mlbench_core.dataset.imagerecognition.pytorch import CIFAR10V1
from mlbench_core.dataset.util.pytorch import partition_dataset_by_rank
from mlbench_core.evaluation.pytorch.metrics import TopKAccuracy
from mlbench_core.models.pytorch.resnet import ResNetCIFAR
from mlbench_core.utils.pytorch import initialize_backends
from mlbench_core.utils.pytorch.checkpoint import CheckpointFreq
from mlbench_core.utils.pytorch.checkpoint import Checkpointer

import numpy as np
import time
import torch
import torch.distributed as dist
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader

# If used in Kubernetes, then comment this line.
os.environ['MLBENCH_IN_DOCKER'] = ""


class SSGDWM(Optimizer):
    # Sparsified SGD with Memory
    # First we only assume we sparsify to top 1
    def __init__(self,
                 model,
                 world_size,
                 num_coordinates,
                 lr=required,
                 weight_decay=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, weight_decay=weight_decay)

        self.model = model
        self.world_size = world_size
        self.num_coordinates = num_coordinates
        super(SSGDWM, self).__init__(model.parameters(), defaults)

        self.__create_gradients_memory()
        self.__create_weighted_average_params()

        # random.seed(0)
        self.rng = np.random.RandomState(100)

    def __create_weighted_average_params(self):
        r""" Create a memory to keep the weighted average of parameters in each iteration """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['estimated_w'] = torch.zeros_like(p.data)
                p.data.normal_(0, 0.01)
                param_state['estimated_w'].copy_(p.data)

    def __create_gradients_memory(self):
        r""" Create a memory to keep gradients that are not used in each iteration """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data.view(-1))

    def cuda(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['estimated_w'] = param_state['estimated_w'].cuda()
                param_state['memory'] = param_state['memory'].cuda()
        return self

    def sparsify_gradients(self, param, lr):
        """ Calls one of the sparsification functions (random or blockwise)

        Args:
            random_sparse (bool): Indicates the way we want to make the gradients sparse
                (random or blockwise) (default: False)
            param (:obj: `torch.nn.Parameter`): Model parameter
        """
        return self._random_sparsify(param, lr)

    def _random_sparsify(self, param, lr):
        """ Sparsify the gradients vector by selecting 'k' of them randomly.

        Args:
            param (:obj: `torch.nn.Parameter`): Model parameter
            lr (float): Learning rate
        """
        grad_view = param.grad.data.view(-1)
        full_size = len(grad_view)

        # Update memory
        self.state[param]['memory'].add_(grad_view * lr)

        # indices of weight to be communicated
        if self.num_coordinates < full_size:
            local_indices = torch.tensor(self.rng.choice(
                full_size,
                self.num_coordinates,
                replace=False))
        else:
            local_indices = torch.tensor(np.arange(full_size))

        # Collect all of the indices for communication
        all_indicies = local_indices

        # Create a sparse vector for the (index, value) of sampled weight
        sparse_tensor = self.state[param]['memory'][all_indicies]
        self.state[param]['memory'][all_indicies] -= sparse_tensor

        return sparse_tensor, all_indicies

    def _aggregate_sparsified_gradients(self):
        for i in range(100):
            for group in self.param_groups:
                lr = group['lr']
                for i, p in enumerate(group['params']):
                    sparse_tensor, random_index = self.sparsify_gradients(
                        p, lr)
                    dist.all_reduce(sparse_tensor, op=dist.reduce_op.SUM)
                    avg_grads = sparse_tensor / self.world_size
                    dv = p.grad.data.view(-1)
                    dv[random_index] = avg_grads

    def _apply_step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            weight_decay = group['weight_decay']

            for p in group['params']:

                if p.grad is None:
                    continue
                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                p.data.add_(-d_p)

        return loss

    def step(self, closure=None):
        # Aggregate with sparsified gradients
        self._aggregate_sparsified_gradients()
        return self._apply_step(closure=closure)


def main(run_id, dataset_dir, ckpt_run_dir, output_dir, validation_only=False):
    r"""Main logic."""
    num_parallel_workers = 2
    use_cuda = True
    max_batch_per_epoch = None
    train_epochs = 164
    batch_size = 128

    initialize_backends(
        comm_backend='mpi',
        logging_level='INFO',
        logging_file=os.path.join(output_dir, 'mlbench.log'),
        use_cuda=use_cuda,
        seed=42,
        cudnn_deterministic=False,
        ckpt_run_dir=ckpt_run_dir,
        delete_existing_ckpts=not validation_only)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    model = ResNetCIFAR(
        resnet_size=20,
        bottleneck=False,
        num_classes=10,
        version=1)

    optimizer = SSGDWM(
        model,
        world_size=world_size,
        num_coordinates=1,
        lr=0.1,
        weight_decay=0)

    # Create a learning rate scheduler for an optimizer
    scheduler = MultiStepLR(
        optimizer,
        milestones=[82, 109],
        gamma=0.1)

    # A loss_function for computing the loss
    loss_function = CrossEntropyLoss()

    if use_cuda:
        model = model.cuda()
        optimizer = optimizer.cuda()
        loss_function = loss_function.cuda()

    # Metrics like Top 1/5 Accuracy
    metrics = [
        TopKAccuracy(topk=1),
        TopKAccuracy(topk=5)
    ]

    train_set = CIFAR10V1(dataset_dir, train=True, download=True)
    val_set = CIFAR10V1(dataset_dir, train=False, download=True)

    train_set = partition_dataset_by_rank(train_set, rank, world_size)
    val_set = partition_dataset_by_rank(val_set, rank, world_size)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False)

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_parallel_workers,
        pin_memory=use_cuda,
        drop_last=False)

    checkpointer = Checkpointer(
        ckpt_run_dir=ckpt_run_dir,
        rank=rank,
        freq=CheckpointFreq.NONE)

    if not validation_only:
        controlflow = TrainValidation(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            metrics=metrics,
            scheduler=scheduler,
            batch_size=batch_size,
            train_epochs=train_epochs,
            rank=rank,
            world_size=world_size,
            run_id=run_id,
            dtype='fp32',
            validate=True,
            schedule_per='epoch',
            checkpoint=checkpointer,
            transform_target_type=None,
            average_models=True,
            use_cuda=use_cuda,
            max_batch_per_epoch=max_batch_per_epoch)

        controlflow.run(
            dataloader_train=train_loader,
            dataloader_val=val_loader,
            dataloader_train_fn=None,
            dataloader_val_fn=None,
            resume=False,
            repartition_per_epoch=False)
    else:
        cecf = CheckpointsEvaluationControlFlow(
            ckpt_dir=ckpt_run_dir,
            rank=rank,
            world_size=world_size,
            checkpointer=checkpointer,
            model=model,
            epochs=train_epochs,
            loss_function=loss_function,
            metrics=metrics,
            use_cuda=use_cuda,
            dtype='fp32',
            max_batch_per_epoch=None)

        train_stats = cecf.evaluate_by_epochs(train_loader)
        with open(os.path.join(output_dir, "train_stats.json"), 'w') as f:
            json.dump(train_stats, f)

        val_stats = cecf.evaluate_by_epochs(val_loader)
        with open(os.path.join(output_dir, "val_stats.json"), 'w') as f:
            json.dump(val_stats, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, default='1',
                        help='The id of the run')
    parser.add_argument('--root-dataset', type=str, default='/datasets',
                        help='Default root directory to dataset.')
    parser.add_argument('--root-checkpoint', type=str, default='/checkpoint',
                        help='Default root directory to checkpoint.')
    parser.add_argument('--root-output', type=str, default='/output',
                        help='Default root directory to output.')
    parser.add_argument('--validation_only', action='store_true',
                        default=False, help='Only validate from checkpoints.')
    args = parser.parse_args()

    uid = 'sparsifiedsgd'
    dataset_dir = os.path.join(args.root_dataset, 'torch', 'cifar10')
    ckpt_run_dir = os.path.join(args.root_checkpoint, uid)
    output_dir = os.path.join(args.root_output, uid)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(ckpt_run_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    main(args.run_id, dataset_dir, ckpt_run_dir,
         output_dir, args.validation_only)
