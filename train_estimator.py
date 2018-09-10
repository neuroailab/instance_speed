from __future__ import division, print_function, absolute_import
import os
import sys
import numpy as np
import tensorflow as tf
import cPickle
import json
import copy
import argparse
import time
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

import utils
import resnet_th_preprocessing as prep_util
import train_instance as train_util


# Function from tensorflow models/official/utils/misc/distribution_utils.py
def get_distribution_strategy(num_gpus, all_reduce_alg=None):
  """Return a DistributionStrategy for running the model.

  Args:
    num_gpus: Number of GPUs to run this model.
    all_reduce_alg: Specify which algorithm to use when performing all-reduce.
      See tf.contrib.distribute.AllReduceCrossTowerOps for available algorithms.
      If None, DistributionStrategy will choose based on device topology.

  Returns:
    tf.contrib.distribute.DistibutionStrategy object.
  """
  if num_gpus == 0:
    return tf.contrib.distribute.OneDeviceStrategy("device:CPU:0")
  elif num_gpus == 1:
    return tf.contrib.distribute.OneDeviceStrategy("device:GPU:0")
  else:
    if all_reduce_alg:
      return tf.contrib.distribute.MirroredStrategy(
          num_gpus=num_gpus,
          cross_tower_ops=tf.contrib.distribute.AllReduceCrossTowerOps(
              all_reduce_alg, num_packs=num_gpus))
    else:
      return tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)


def model_function(
        features,
        labels,
        mode,
        params,
        ):
    assert mode != tf.estimator.ModeKeys.PREDICT, \
            "Predict mode not implemented!"
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    use_tpu = params['use_tpu']
    model_params = {
            'inputs': features,
            'use_tpu': use_tpu,
            }
    train_output = utils.build_instance_model(
            train=is_training,
            **model_params
            )
    loss_all, _, _ = train_util.get_loss_all(
            train_output,
            train_util.data_len, 
            params['weight_decay'])

    # Build train_op here
    if is_training:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = params['init_lr']

        optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=0.9
                )
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_all, global_step=global_step)
    else:
        train_op = None

    # Build evaluation metrics here
    if is_training:
        scope = tf.get_variable_scope()
        scope.reuse_variables()
        val_output = utils.build_instance_model(
                train=False,
                **model_params
                )
    else:
        val_output = train_output
    def _metric_fn(*args):
        curr_dist = args[0]
        if not use_tpu:
            all_labels = args[2]
            curr_labels = labels
        else:
            if not params['use_synth']:
                all_labels = tf.constant(
                        cPickle.load(open(params['instance_lbl_pkl'], 'r')),
                        dtype=tf.int64,
                        )
            else:
                all_labels = tf.constant(
                        np.zeros((train_util.data_len)),
                        dtype=tf.int64,
                        )
            curr_labels = args[2]
        _, top_indices = tf.nn.top_k(curr_dist, k=1)
        curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
        imagenet_top1 = tf.metrics.mean(
                tf.cast(
                    tf.equal(curr_pred, curr_labels),
                    tf.float32))
        return {'top1': imagenet_top1}

    if not use_tpu:
        metrics = _metric_fn(*val_output)
    else:
        metrics = (_metric_fn, val_output)

    kwargs = {
            'mode':mode,
            'predictions':val_output[0],
            'loss':loss_all,
            'train_op':train_op,
            }
    if not use_tpu:
        estimator_fn = tf.estimator.EstimatorSpec
        kwargs['eval_metric_ops'] = metrics
    else:
        estimator_fn = tpu_estimator.TPUEstimatorSpec
        kwargs['eval_metrics'] = metrics

    return estimator_fn(
            **kwargs
            )


# Get parser from train_util, and add estimator/tpu related arguments
def get_parser():
    parser = train_util.get_parser()
    parser.add_argument(
            '--valid_first', 
            action='store_true', help='Whether validating first')
    parser.add_argument(
	    '--inst_lbl_pkl', default=None, type=str, action='store',
            help='Label pkl file for instance task')
    parser.add_argument(
	    '--tpu_name', default=None, type=str, action='store',
            help='Tpu name, None means using usual estimator')
    parser.add_argument(
	    '--gcp_project', default=None, type=str, action='store',
            help='Project id')
    parser.add_argument(
	    '--tpu_zone', default=None, type=str, action='store',
            help='Tpu zone name')
    parser.add_argument(
            '--use_synth', 
            action='store_true', help='Whether using synthetic data')
    return parser


def get_synth_data(batch_size, with_indx=True):
    synth_image = tf.random_uniform(
            shape=(batch_size, 224, 224, 3),
            minval=0,
            maxval=255,
            )
    synth_label = tf.random_uniform(
            shape=(batch_size,),
            minval=0,
            maxval=train_util.num_cat,
            dtype=tf.int64,
            )
    if with_indx:
        synth_indx = tf.random_uniform(
                shape=(batch_size,),
                minval=0,
                maxval=train_util.data_len,
                dtype=tf.int64,
                )
    dict_dataset = {
            'image': tf.data.Dataset.from_tensors(synth_image).repeat(),
            'label': tf.data.Dataset.from_tensors(synth_label).repeat(),
            }
    if with_indx:
        dict_dataset['index'] = tf.data.Dataset.\
                from_tensors(synth_indx).repeat()
    dataset = tf.data.Dataset.zip(dict_dataset)
    dataset = tf.data.Dataset.zip((
        dataset,
        tf.data.Dataset.from_tensors(synth_label).repeat(),
        ))
    return dataset


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_tpu = args.tpu_name!=None

    # Set number of steps in each epoch of train/validation
    num_step_one_epoch = int(train_util.data_len/args.batch_size)
    num_val_step = int(train_util.val_len/args.test_batch_size)

    # Using the Winograd non-fused algorithms to get a small boost
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    tf.logging.set_verbosity(tf.logging.INFO)

    model_dir = os.path.join(args.cache_dir_prefix, args.exp_id)
    if not use_tpu:
        os.system('mkdir -p %s' % model_dir)

    model_params = {
            'weight_decay': args.weight_decay,
            'init_lr': args.init_lr,
            'use_tpu': use_tpu,
            'instance_lbl_pkl': args.inst_lbl_pkl,
            'use_synth': args.use_synth,
            }
    if not use_tpu:
        #distribution_strategy = get_distribution_strategy(
        #        num_gpus=len(args.gpu.split(',')))
        session_config = tf.ConfigProto(allow_soft_placement=True)
        run_config = tf.estimator.RunConfig(
                #train_distribute=distribution_strategy, 
                session_config=session_config,
                save_checkpoints_steps=num_step_one_epoch,
                )
        # Build the estimator
        classifier = tf.estimator.Estimator(
                model_fn=model_function, model_dir=model_dir,
                config=run_config, params=model_params)
    else:
        tpu_cluster_resolver = (
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    tpu=[args.tpu_name],
                    zone=args.tpu_zone,
                    project=args.gcp_project))
        tpu_grpc_url = tpu_cluster_resolver.get_master()
        run_config = tpu_config.RunConfig(
                master=tpu_grpc_url,
                evaluation_master=tpu_grpc_url,
                model_dir=model_dir,
                save_checkpoints_steps=num_step_one_epoch,
                save_checkpoints_secs=None,
                log_step_count_steps=100,
                tpu_config=tpu_config.TPUConfig(
                    iterations_per_loop=100,
                    num_shards=8))
        classifier = tpu_estimator.TPUEstimator(
                use_tpu=True,
                model_fn=model_function,
                config=run_config,
                train_batch_size=args.batch_size,
                eval_batch_size=args.test_batch_size,
                params=model_params)

    train_hooks = None

    def input_fn_train(num_epochs, params={}):
        batch_size = params.get('batch_size', args.batch_size)

        if not args.use_synth:
            # Get tfrecords file names
            train_tfr_list = train_util.get_tfr_filenames(args.image_dir)
            np.random.seed(num_epochs)
            curr_shuffle = np.random.permutation(len(train_tfr_list))
            list_file_dataset = tf.data.Dataset.list_files(
                    train_tfr_list[curr_shuffle])
            dataset = list_file_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        train_util.fetch_dataset, cycle_length=8, sloppy=True))

            # parse and preproecess it 
            train_dataset = dataset.shuffle(buffer_size=51200).map(
                    lambda x: train_util.data_paser(
                        x, True, 
                        with_indx=True,
                        ret_float=use_tpu), 
                    num_parallel_calls=64)

            # batch it
            train_data = train_util.batch_iterator(
                    train_dataset, 
                    batch_size,
                    one_shot=True)
            return train_data, train_data['label']
        else:
            train_dataset = get_synth_data(batch_size)
            return train_dataset

    def input_fn_eval(params={}):
        batch_size = params.get('batch_size', args.test_batch_size)
        
        if not args.use_synth:
            val_tfr_list = train_util.get_tfr_filenames(
                    args.image_dir, 
                    'validation-*')
            list_file_dataset = tf.data.Dataset.list_files(
                    val_tfr_list)
            dataset = list_file_dataset.apply(
                    tf.contrib.data.parallel_interleave(
                        train_util.fetch_dataset, cycle_length=8, sloppy=True))
            val_dataset = dataset.map(
                    lambda x: train_util.data_paser(
                        x, False, 
                        with_indx=False,
                        ret_float=use_tpu), 
                    num_parallel_calls=64)

            val_data = train_util.batch_iterator(
                    val_dataset, 
                    batch_size,
                    one_shot=True)
            return val_data, val_data['label']
        else:
            val_dataset = get_synth_data(batch_size, with_indx=False)
            return val_dataset

    if args.valid_first:
        tf.logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(
                input_fn=input_fn_eval,
                steps=num_val_step)
    for num_train_epochs in range(200):
        classifier.train(
                input_fn=lambda params: \
                        input_fn_train(num_train_epochs, params),
                steps=num_step_one_epoch)
        tf.logging.info('Starting to evaluate.')
        eval_results = classifier.evaluate(
                input_fn=input_fn_eval,
                steps=num_val_step)


if __name__ == "__main__":
    main()
