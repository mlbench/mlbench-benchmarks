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


def define_ops_to_run(model, data_loader, is_training):
    # intermediate ops
    logits = model(data_loader.inputs, is_training)

    def loss_filter_fn(name):
        """Filter trainable variables with batch_normalization."""
        return 'batch_normalization' not in name

    loss = softmax_cross_entropy_with_logits_v2_l2_regularized(
        logits=logits,
        labels=data_loader.labels,
        l2=2e-4,
        loss_filter_fn=loss_filter_fn)

    # Use Top K accuracy as metrics
    metrics = [
        topk_accuracy_with_logits(logits, data_loader.labels, k=1),
        topk_accuracy_with_logits(logits, data_loader.labels, k=5),
    ]

    # Define a global_step op which counts the number times gradients have been applied.
    global_step = tf.train.get_or_create_global_step()

    epoch = tf.Variable(0, name='epoch')
    lr_scheduler = manual_stepping(
        global_step=epoch,
        boundaries=[82, 109],
        rates=[0.1, 0.01, 0.001],
        warmup=False)

    # Define the optimizer
    optimizer = tf.train.MomentumOptimizer(
        learning_rate=lr_scheduler,
        momentum=0.9,
        use_nesterov=True
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

    # raise NotImplementedError((train_op), type(train_op))
    return train_op, loss, metrics, epoch


def run(sess):
    world_size = 1
    batch_size = 128
    model = Cifar10Model(resnet_size=20,
                         data_format='channels_last',
                         resnet_version=2,
                         dtype=tf.float32)

    data_loader = DatasetCifar(
        dataset='cifar-10',
        dataset_root='/datasets',
        batch_size=batch_size,
        world_size=world_size,
        seed=42,
        tf_dtype=tf.float32)

    is_training = tf.placeholder(tf.bool, (), name='is_training')

    train_op, loss, metrics, epoch = define_ops_to_run(
        model, data_loader, is_training)

    # The placeholders here will be used in the `sess.run`
    cf = ControlFlow(is_training=is_training,
                     train_op=train_op,
                     data_loader=data_loader,
                     sess=sess,
                     loss=loss,
                     metrics=metrics,
                     lr_scheduler_level='epoch',
                     max_train_steps=164,
                     train_epochs=164)

    cf.train_and_eval(lr_scheduler=epoch)


def main():
    initialize_backends(None)

    sess_config = default_session_config(
        tf_allow_soft_placement=False,
        tf_log_device_placement=False,
        tf_gpu_mem=0.1)

    with tf.Graph().as_default():
        sess = tf.Session(config=sess_config)
        with sess.as_default():
            run(sess)


if __name__ == '__main__':
    main()
