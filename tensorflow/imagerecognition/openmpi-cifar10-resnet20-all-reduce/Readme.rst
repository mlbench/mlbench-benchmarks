Tensorflow Cifar-10 ResNet-20 Open-MPI
""""""""""""""""""""""""""""""""""""""

.. TODO We use OpenMPI for starting processes, but communication is gRPC? document this more
   Also, All_Reduce is not exactly all-reduce. We need to comment on this

:Framework: TensorFlow
:Communication Backend: Open MPI
:Distribution Algorithm: All-Reduce
:Model: ResNet-20
:Dataset: CIFAR-10
:GPU: Yes
:Seed: 42
:Image Location: /tensorflow/imagerecognition/openmpi-cifar10-resnet20-all-reduce/