r"""Create a Resnet Model and Output something."""
import types
import tensorflow as tf

from mlbench_core.utils.tensorflow import initialize_backends, default_session_config
from mlbench_core.models.tensorflow.resnet_model import Cifar10Model
from mlbench_core.dataset.imagerecognition.tensorflow.cifar10 import DatasetCifar
from mlbench_core.controlflow.tensorflow.train_and_eval import ControlFlow
from mlbench_core.lr_scheduler.tensorflow.lr import manual_stepping
from mlbench_core.evaluation.tensorflow.metrics import topk_accuracy_with_logits
from mlbench_core.evaluation.tensorflow.criterion import \
    softmax_cross_entropy_with_logits_v2_l2_regularized


def define_ops_to_run(config, model, data_loader, is_training):
    # intermediate ops
    logits = model(data_loader.inputs, is_training)

    def loss_filter_fn(name):
        """Filter trainable variables with batch_normalization."""
        return 'batch_normalization' not in name if not config.l2_reg_bn else True

    loss = softmax_cross_entropy_with_logits_v2_l2_regularized(
        logits, data_loader.labels, config.l2, loss_filter_fn)

    # Use Top K accuracy as metrics
    metrics = [
        topk_accuracy_with_logits(logits, data_loader.labels, k=k)
        for k in [1, 5]
    ]

    # Define a global_step op which counts the number times gradients have been applied.
    global_step = tf.train.get_or_create_global_step()

    epoch = tf.Variable(0, name='epoch')
    lr_scheduler = manual_stepping(
        global_step=epoch, boundaries=[82, 109], rates=[0.1, 0.01, 0.001], warmup=False)

    # Define the optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=lr_scheduler,
        momentum=config.momentum,
        use_nesterov=config.nesterov
    )

    # The update_ops is needed if batch normalization is used.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         # Compute gradients
        grads_and_vars = list(optimizer.compute_gradients(
            loss, tf.trainable_variables()))

        minimize_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        train_op = tf.group(minimize_op)

    return train_op, loss, metrics, epoch


def run(config, sess):
    model = Cifar10Model(config.resnet_size,
                         config.tf_data_format,
                         resnet_version=config.resnet_version,
                         dtype=config.tf_dtype)

    data_loader = DatasetCifar(config)

    is_training = tf.placeholder(tf.bool, (), name='is_training')

    train_op, loss, metrics, epoch = define_ops_to_run(
        config, model, data_loader, is_training)

    # The placeholders here will be used in the `sess.run`
    cf = ControlFlow(is_training=is_training,
                     train_op=train_op,
                     data_loader=data_loader,
                     sess=sess,
                     config=config,
                     loss=loss,
                     metrics=metrics)

    cf.train_and_eval(lr_scheduler=epoch)


def main(config):
    initialize_backends(config)

    sess_config = default_session_config(config)

    with tf.Graph().as_default():
        sess = tf.Session(config=sess_config)
        with sess.as_default():
            run(config, sess)


if __name__ == '__main__':
    DATASET_CONFIG = {
        "dataset": "cifar-10",
        "dataset_version": 1,
        "dataset_root": "/datasets",
        "num_parallel_workers": 2,
        "batch_size": 256,
        "shuffle_before_partition": True,
        "num_classes": 10
    }

    OPTIMIZER_CONFIG = {
        "momentum": 0.9,
        "nesterov": True,

        "lr_scheduler_level": "epoch"
    }

    MOCK_CONFIG = {
        "world_size": 1,
        "rank": 0,
        "seed": 42,
    }

    TF_LEGACY_CONFIG = {
        "tf_gpu_thread_mode": True,
        "inter_op_parallelism_threads": 2,
        "intra_op_parallelism_threads": 2,

        # 1 if we use tf.float32 and 128 if we use tf.float16
        "tf_loss_scale": 1,

        "tf_dtype": tf.float32,

        "tf_data_format": "channels_last",
        "resnet_size": 20,
        "resnet_version": 1,

        "train_epochs": 164,
        "max_train_steps": 164,
        "model_dir": "/checkpoint/tmp",
        "eval_only": False,
        "lr": 0.1,

        "data_dir": "/datasets",
        "datasets_num_private_threads": 1,
        "export_dir": None,
        "datasets_num_parallel_batches": 2
    }

    TF_CONFIG = {
        "tf_log_device_placement": False,
        "tf_allow_soft_placement": False,
        "tf_gpu_mem": 0.1,
    }

    TMP_CONFIG = {
        "l2_reg_bn": True,
        "l2": 2e-4
    }

    config = types.SimpleNamespace(**DATASET_CONFIG, **TF_CONFIG, **TMP_CONFIG,
                                   **MOCK_CONFIG, **TF_LEGACY_CONFIG, **OPTIMIZER_CONFIG)
    main(config)
