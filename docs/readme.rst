.. |br| raw:: html

    <br>

MLBench Benchmarks
==================

MLBench contains several benchmark tasks and implementations.

Benchmark Tasks
---------------
A Benchmark Task is a specific dataset and evaluation task, e.g. the CIFAR-10 image dataset with the task of classifying objects in the images.


Image Recognition
~~~~~~~~~~~~~~~~~

CIFAR-10
++++++++

The `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__ dataset contains 60000 32x32 colour images in 10 classes, such as ``airplane`` or ``dog``, with 6000 images per class.
There are 50000 training images and 10000 test images.

The task is to correctly predict what class each image belongs to.

The evaluation criterion is mean accuracy of predictions over all test images



Benchmark Implementations
-------------------------

A Benchmark Implementation is a model with fixed hyperparameters that solves a Benchmark Task.


Image Recognition
~~~~~~~~~~~~~~~~~


CIFAR-10
++++++++

.. table:: CIFAR-10 Benchmark Implementations. TtA @ 1, 2, 4, 8, 16 = Time to Accuracy with 1, 2, 4, 8, 16 workers. AaT 2h @ 1, 2, 4, 8, 16 = Accuracy after Time (2 Hours) with 1, 2, 4, 8, 16 workers
   :widths: auto

   +-------------------------------------+-----------+---------+-------------+-----------+------------------------------+------------------------------+--------------------------------------------+
   | Name                                | Framework | Backend | Parallelism | Model     | TtA @ 1, 2, 4, 8, 16         | AaT 2h @ 1, 2, 4, 8, 16      | Details                                    |
   +=====================================+===========+=========+=============+===========+==============================+==============================+============================================+
   | openmpi-cifar10-resnet18-all-reduce | PyTorch   | OpenMPI | All-Reduce  | ResNet-18 | 1: XX.X  |br|                | 1: XX.X |br|                 | :doc:`openmpi-cifar10-resnet18-all-reduce` |
   |                                     |           |         |             |           | 2: XX.X  |br|                | 2: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 4: XX.X  |br|                | 4: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 8: XX.X  |br|                | 8: XX.X |br|                 |                                            |
   |                                     |           |         |             |           | 16: XX.X |br|                | 16: XX.X  |br|               |                                            |
   +-------------------------------------+-----------+---------+-------------+-----------+------------------------------+------------------------------+--------------------------------------------+


ImageNet
++++++++

[.....]



All Implementations
-------------------

.. toctree::

   openmpi-cifar10-resnet18-all-reduce