from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np

import tensorflow as tf
import cPickle

import json
import copy
import argparse
import time

import utils
import resnet_th_preprocessing as prep_util


## Set length of dataset manually
data_len = 1281167
val_len = 50000
num_cat = 1000


def get_parser():
    parser = argparse.ArgumentParser(
            description='Train instance task using dataset interface')
    parser.add_argument(
            '--exp_id', 
            default="instance", type=str, 
            action='store', help='Name of experiment id')
    parser.add_argument(
            '--cache_dir_prefix', 
            default="/mnt/fs1/chengxuz/notf_model", 
            type=str, action='store', 
            help='Prefix of cache directory')
    parser.add_argument(
            '--batch_size', 
            default=128, 
            type=int, action='store', help='Batch size')
    parser.add_argument(
            '--test_batch_size', 
            default=64, 
            type=int, action='store', help='Batch size')
    parser.add_argument(
            '--init_lr', 
            default=0.03, type=float, 
            action='store', help='Init learning rate')
    parser.add_argument(
            '--gpu', 
            default='0', type=str, 
            action='store', help='Availabel GPUs')
    parser.add_argument(
            '--weight_decay', 
            default=1e-4, type=float, 
            action='store', help='Weight decay')
    parser.add_argument(
            '--restore_path', 
            default=None, type=str, 
            action='store', help='Path to saved file')
    parser.add_argument(
            '--image_dir', 
            default="/mnt/fs1/Dataset/TFRecord_Imagenet_standard/"\
                    + "image_label_full_widx", 
            type=str, action='store', help='Prefix of cache directory')
    return parser


def get_tfr_filenames(folder_name, file_pattern='train-*'):
    '''
    Get list of tfrecord filenames 
    for given folder_name fitting the given file_pattern
    '''
    tfrecord_pattern = os.path.join(folder_name, file_pattern)
    datasource = tf.gfile.Glob(tfrecord_pattern)
    datasource.sort()
    return np.asarray(datasource)


# Useful util function
def fetch_dataset(filename):
    buffer_size = 32 * 1024 * 1024     # 32 MiB per file
    dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
    return dataset


def data_paser(
        value, is_train=True,
        with_indx=True,
        ret_float=False,
        ):
    # Load the image and preprocess it
    keys_to_features = {
            'images': tf.FixedLenFeature((), tf.string, ''),
            'labels': tf.FixedLenFeature([], tf.int64, -1)}
    if with_indx:
        keys_to_features['index'] = tf.FixedLenFeature([], tf.int64, -1)
    parsed = tf.parse_single_example(value, keys_to_features)
    image_string = parsed['images']
    image_label = parsed['labels']
    image_index = parsed.get('index', None)

    # Do the preprocessing
    crop_size = 224
    if is_train:
        image_crop = prep_util.RandomSizedCrop_from_jpeg(
                image_string,
                out_height=crop_size,
                out_width=crop_size,
                size_minval=0.2,)
        image_rdGray = prep_util.ApplyGray(
                image_crop, 0.2)
        image_color = prep_util.ColorJitter(
                image_rdGray)
        image = tf.image.random_flip_left_right(image_color)
    else:
        image_decode = tf.image.decode_image(image_string, channels=3)
        image = prep_util._aspect_preserving_resize(image_decode, 256)
        image = prep_util._central_crop([image], crop_size, crop_size)[0]
        image.set_shape([crop_size, crop_size, 3])
    if ret_float:
        image = tf.cast(image, tf.float32)
    ret_dict = {
            'image':image, 
            'label':image_label}
    if with_indx:
        ret_dict['index'] = image_index
    return ret_dict


def batch_iterator(dataset, batch_size, one_shot=False):
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.apply(
        tf.contrib.data.batch_and_drop_remainder(batch_size))
    dataset = dataset.prefetch(2)
    if not one_shot:
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        return iterator, next_element
    else:
        next_element = dataset.make_one_shot_iterator().get_next()
        return next_element


def get_loss_all(train_output, data_len, weight_decay):
    loss_pure, loss_model, loss_noise = utils.instance_loss(
            train_output[0], train_output[1],
            instance_k=4096,
            instance_data_len=data_len,)
    # Add weight decay to the loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    l2_loss = weight_decay * tf.add_n(
            # loss is computed using fp32 for numerical stability.
            [tf.nn.l2_loss(tf.cast(v, tf.float32)) 
                for v in tf.trainable_variables() 
                if exclude_batch_norm(v.name)])
    loss_all = tf.add(loss_pure, l2_loss)
    return loss_all, loss_model, loss_noise


def main():
    # Parse arguments
    parser = get_parser()
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # Get dataset
    ## Build list file dataset
    train_tfr_list = get_tfr_filenames(args.image_dir)
    val_tfr_list = get_tfr_filenames(args.image_dir, 'validation-*')
    list_file_place = tf.placeholder(tf.string)
    list_file_dataset = tf.data.Dataset.list_files(list_file_place)
    dataset = list_file_dataset.apply(
            tf.contrib.data.parallel_interleave(
                fetch_dataset, cycle_length=8, sloppy=True))
    train_dataset = dataset.shuffle(buffer_size=51200).map(
            lambda x: data_paser(
                x, True, 
                with_indx=True), 
            num_parallel_calls=64)
    val_dataset = dataset.map(
            lambda x: data_paser(
                x, False, 
                with_indx=False), 
            num_parallel_calls=64)
    ## Get iterators
    train_iterator, train_next_element = batch_iterator(
            train_dataset, 
            args.batch_size)
    val_iterator, val_next_element = batch_iterator(
            val_dataset, 
            args.test_batch_size)

    # Build network
    model_params = {}
    train_output = utils.build_instance_model(
            inputs=train_next_element,
            train=True,
            **model_params)
    global_step = tf.get_variable(
            'global_step', [],
            dtype=tf.int64, trainable=False,
            initializer=tf.constant_initializer(0))
    global_epoch = tf.get_variable(
            'global_epoch', [],
            dtype=tf.int64, trainable=False,
            initializer=tf.constant_initializer(0))
    epoch_update_op = tf.assign(global_epoch, global_epoch+1)

    # Get loss and optimizer
    loss_all, loss_model, loss_noise = get_loss_all(
            train_output, 
            data_len, 
            args.weight_decay)
    optimizer = tf.train.MomentumOptimizer(
            learning_rate=args.init_lr, 
            momentum=0.9)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        mini_act = optimizer.minimize(loss_all, global_step=global_step)
    ## For validation
    scope = tf.get_variable_scope()
    scope.reuse_variables()
    val_output = utils.build_instance_model(
            inputs=val_next_element,
            train=False,
            **model_params)
    curr_dist, _, all_labels = val_output
    _, top_indices = tf.nn.top_k(curr_dist, k=1)
    curr_pred = tf.gather(all_labels, tf.squeeze(top_indices, axis=1))
    imagenet_top1 = tf.reduce_mean(
            tf.cast(
                tf.equal(curr_pred, val_next_element['label']), 
                tf.float32))

    # Get session
    saver = tf.train.Saver(tf.global_variables())
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=gpu_options,
            ))
    ## Run initialization
    init_op_global = tf.global_variables_initializer()
    sess.run(init_op_global)
    init_op_local = tf.local_variables_initializer()
    sess.run(init_op_local)

    # Train and validation
    ## Logging related
    os.system('mkdir -p %s' % os.path.join(args.cache_dir_prefix, args.exp_id))
    log_perf_name = os.path.join(
            args.cache_dir_prefix, 
            'log_performance_%s.txt' % args.exp_id)
    log_perf_writer = open(log_perf_name, 'a+')
    ## Preparation
    if args.restore_path is not None:
        saver.restore(sess, args.restore_path)
    num_step_one_epoch = int(data_len/args.batch_size)
    num_val_step = int(val_len/args.test_batch_size)
    sta_epoch = sess.run(global_epoch)

    for curr_epoch in range(sta_epoch, 200):
        ## Training
        np.random.seed(curr_epoch)
        curr_shuffle = np.random.permutation(len(train_tfr_list))
        sess.run(train_iterator.initializer, feed_dict={
                list_file_place:train_tfr_list[curr_shuffle],
                })
        all_time = []
        for curr_step in range(num_step_one_epoch):
            gpu_stime = time.time()
            _, value_all, value_model, value_noise = sess.run(
                    [mini_act, loss_all, loss_model, loss_noise])
            gpu_time = time.time() - gpu_stime
            if curr_step>0:
                all_time.append(gpu_time)
            perf_str = ("Epoch %i[%i/%i]: Loss: %.4f, Loss model: %.4f,"\
                    + " Loss noise: %.4f, Time: %.3f (%.3f)") \
                    % (curr_epoch, curr_step, num_step_one_epoch, \
                    value_all, value_model, value_noise, \
                    gpu_time, np.mean(all_time))
            print(perf_str)
            sys.stdout.flush()
            log_perf_writer.write(perf_str + '\n')
            if (curr_step+1) % 200==0:
                log_perf_writer.close()
                log_perf_writer = open(log_perf_name, 'a+')
        ## Validation
        sess.run(val_iterator.initializer, feed_dict={
                list_file_place:val_tfr_list,
                })
        all_top1 = []
        for curr_step in range(num_val_step):
            curr_top1 = sess.run(imagenet_top1)
            all_top1.append(curr_top1)
            perf_str = "Test [%i/%i]: Top1: %.4f, Ave Top1: %.4f" \
                    % (curr_step, num_val_step, curr_top1, np.mean(all_top1))
            print(perf_str)
            sys.stdout.flush()
            log_perf_writer.write(perf_str + '\n')
            if (curr_step+1) % 200==0:
                log_perf_writer.close()
                log_perf_writer = open(log_perf_name, 'a+')
        sess.run(epoch_update_op)
        saver.save(
                sess, 
                os.path.join(args.cache_dir_prefix, args.exp_id, 'model.ckpt'), 
                global_step=sess.run(global_step))

    log_perf_writer.close()

if __name__ == "__main__":
    main()
