r"""Distributed TensorFlow with Monitored Training Session.

Adapted from official tutorial::

    https://www.tensorflow.org/deploy/distributed

Launch::

    mpirun -n 3 --allow-run-as-root python ....

"""
import argparse
import logging
import os
import tensorflow as tf


from mlbench_core.utils.tensorflow import initialize_backends, default_session_config
from mlbench_core.models.tensorflow.resnet_model import Cifar10Model
from mlbench_core.dataset.imagerecognition.tensorflow.cifar10 import DatasetCifar
from mlbench_core.lr_scheduler.tensorflow.lr import manual_stepping
from mlbench_core.evaluation.tensorflow.metrics import topk_accuracy_with_logits
from mlbench_core.evaluation.tensorflow.criterion import \
    softmax_cross_entropy_with_logits_v2_l2_regularized
from mlbench_core.controlflow.tensorflow.train_validation import TrainValidation


def define_graph(inputs, labels, is_training, batch_size, replicas_to_aggregate):
    """
    Define graph for synchronized training.
    """
    model = Cifar10Model(
        resnet_size=20,
        data_format='channels_last',
        resnet_version=2,
        dtype=tf.float32)

    logits = model(inputs, is_training)

    loss = softmax_cross_entropy_with_logits_v2_l2_regularized(
        logits=logits,
        labels=labels,
        l2=2e-4,
        # Exclude BN weights from L2 regularizer
        loss_filter_fn=lambda name: 'batch_normalization' not in name)

    # Use Top K accuracy as metrics
    metrics = [
        topk_accuracy_with_logits(logits, labels, k=1),
        topk_accuracy_with_logits(logits, labels, k=5),
    ]

    global_step = tf.train.get_or_create_global_step()

    # scheduling learning steps.
    lr_scheduler = manual_stepping(
        global_step=global_step,
        boundaries=[32000 // replicas_to_aggregate,
                    48000 // replicas_to_aggregate],
        rates=[0.1, 0.01, 0.001],
        warmup=False)

    # Define the optimizer
    optimizer_ = tf.train.MomentumOptimizer(
        learning_rate=lr_scheduler,
        momentum=0.9,
        use_nesterov=True)

    # Wrap optimizer with `SyncReplicasOptimizer`
    optimizer = tf.train.SyncReplicasOptimizer(
        optimizer_,
        replicas_to_aggregate=replicas_to_aggregate,
        total_num_replicas=replicas_to_aggregate)

    hooks = [
        optimizer.make_session_run_hook((rank == 0), num_tokens=0)
    ]

    # The update for batch normalization.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        # Not all of the processes contribute one update. Some faster procs can push more updates.
        grads_and_vars = list(optimizer.compute_gradients(
            loss, tf.trainable_variables()))

        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    return train_op, loss, metrics, hooks


def main(is_ps, run_id, rank, world_size, cluster_spec, batch_size,
         replicas_to_aggregate):
    logging.info("Initial.")

    job_name = "ps" if is_ps else "worker"

    cluster = tf.train.ClusterSpec(cluster_spec)

    gpu_options = tf.GPUOptions(allow_growth=True,
                                per_process_gpu_memory_fraction=0.2)

    session_conf = tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True,
        log_device_placement=False)

    server = tf.train.Server(
        cluster, job_name=job_name, task_index=rank, config=session_conf)

    if is_ps:
        server.join()
    else:
        # Pin variables to parameter server.
        device_fn = tf.train.replica_device_setter(
            ps_tasks=None,
            ps_device="/job:ps",
            worker_device="/job:{}/task:{}/device:GPU:{}".format(
                job_name, rank, rank),
            merge_devices=True,
            cluster=cluster,
            ps_ops=None,
            ps_strategy=None)

        with tf.Graph().as_default():
            with tf.device(device_fn):
                data_loader = DatasetCifar(
                    dataset='cifar-10',
                    dataset_root='/datasets',
                    batch_size=batch_size,
                    world_size=world_size,
                    rank=rank,
                    seed=42,
                    tf_dtype=tf.float32)

                train_op, loss, metrics, hooks = define_graph(
                    data_loader.inputs,
                    data_loader.labels,
                    data_loader.training,
                    batch_size,
                    replicas_to_aggregate)

                local_init_op = tf.group(
                    tf.local_variables_initializer(),
                    data_loader.train_init_op,
                    data_loader.validation_init_op)

                scaffold = tf.train.Scaffold(
                    init_op=None,
                    init_feed_dict=None,
                    init_fn=None,
                    ready_op=None,
                    ready_for_local_init_op=None,
                    local_init_op=local_init_op)

                lr_tensor_name = tf.get_default_graph().get_tensor_by_name("learning_rate:0")

            with tf.train.MonitoredTrainingSession(config=session_conf,
                                                   master=server.target,
                                                   scaffold=scaffold,
                                                   is_chief=(rank == 0),
                                                   checkpoint_dir=None,
                                                   save_checkpoint_secs=None,
                                                   save_summaries_steps=None,
                                                   stop_grace_period_secs=5,
                                                   hooks=hooks) as sess:

                logging.info("Begin training.")

                cf = TrainValidation(
                    batch_size=batch_size,
                    train_set_init_op=data_loader.train_init_op,
                    validation_set_init_op=data_loader.validation_init_op,
                    num_batches_per_epoch_for_train=data_loader.num_batches_per_epoch_for_train,
                    num_batches_per_epoch_for_validation=data_loader.num_batches_per_epoch_for_eval,
                    train_op=train_op,
                    sess=sess,
                    loss=loss,
                    metrics=metrics,
                    lr_scheduler_level='epoch',
                    max_train_steps=164,
                    train_epochs=164,
                    run_id=run_id,
                    rank=rank)

                cf.train_and_eval(lr_tensor_name=lr_tensor_name)

            logging.info("Finish.")


def configure_logger(log_dir, is_ps, rank):
    logger = logging.getLogger('')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        '{:6} rank={} : %(message)s'.format("ps" if is_ps else "worker", rank),
        "%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    log_name = '{}-{}.log'.format("ps" if is_ps else "worker", rank)
    log_name = os.path.join(log_dir, log_name)
    if os.path.exists(log_name):
        os.remove(log_name)

    fh = logging.FileHandler(log_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process run parameters')
    parser.add_argument('--run_id', type=str, help='The id of the run')
    parser.add_argument('--hosts', type=str, help='The hosts participating in this run')
    args = parser.parse_args()

    rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    size = int(os.environ['OMPI_COMM_WORLD_SIZE'])

    hosts = args.hosts.split(",")

    if len(hosts) < 2:
        raise ValueError("At least 2 pods are needed for this benchmark (1 parameter server, 1 worker)")

    workers = [h + ":22222" for h in hosts[1:]]
    ps = hosts[0] + ":22222" # First worker is the parameter server

    cluster_spec = {"worker": workers,
                    "ps": [ps]}

    # Parse role in the cluster by rank.
    is_ps = rank < len(cluster_spec['ps'])
    rank = rank if is_ps else rank - len(cluster_spec['ps'])
    world_size = size - len(cluster_spec['ps'])

    # Configure Logging
    if not os.path.exists('/mlbench'):
        os.makedirs('/mlbench')
    configure_logger('/mlbench', is_ps, rank)

    batch_size = 128
    replicas_to_aggregate = len(cluster_spec['worker'])

    main(is_ps, args.run_id, rank, world_size, cluster_spec,
         batch_size, replicas_to_aggregate)
