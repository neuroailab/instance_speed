from __future__ import division, print_function, absolute_import
import os, sys
import numpy as np
import tensorflow as tf
import copy
import pdb
import resnet_model
from resnet_th_preprocessing import ColorNormalize
import cPickle


def instance_loss(
        data_dist, noise_dist, 
        instance_k, instance_data_len
        ):
    base_prob = 1.0 / instance_data_len
    eps = 1e-7
    ## Pmt
    data_div = data_dist + (instance_k*base_prob + eps)
    ln_data = tf.log(data_dist / data_div)
    ## Pon
    noise_div = noise_dist + (instance_k*base_prob + eps)
    ln_noise = tf.log((instance_k*base_prob) / noise_div)

    batch_size = data_dist.get_shape().as_list()[0]
    curr_loss = -(tf.reduce_sum(ln_data) \
            + tf.reduce_sum(ln_noise)) / batch_size

    return curr_loss, \
            -tf.reduce_sum(ln_data)/batch_size, \
            -tf.reduce_sum(ln_noise)/batch_size


def build_instance_model(
        inputs,
        train,
        resnet_size=18,
        data_format=None,
        resnet_version=resnet_model.DEFAULT_VERSION,
        dtype=tf.float32,
        instance_k=4096,
        instance_t=0.07,
        instance_m=0.5,
        instance_data_len=1281167,
        model_seed=0,
        use_tpu=False,
        ):
    input_name = 'image'
    image = tf.cast(inputs[input_name], tf.float32)
    image = tf.div(image, tf.constant(255, dtype=tf.float32))
    image = tf.map_fn(ColorNormalize, image)

    # Build resnet model here, the only difference is that the output is 128-D
    model = resnet_model.ImagenetModel(
            resnet_size, data_format, 
            resnet_version=resnet_version,
            dtype=dtype)
    curr_out = model(image, train)
        
    # Build memory bank
    curr_out = tf.nn.l2_normalize(curr_out, axis=1) # [bs, out_dim]
    all_outputs = []
    ## Actually define the memory bank (and label bank)
    with tf.variable_scope('instance', reuse=tf.AUTO_REUSE):
        ### Set the variable and initial values
        batch_size, out_dim = curr_out.get_shape().as_list()
        if not use_tpu:
            mb_init = tf.random_uniform(
                    shape=(instance_data_len, out_dim),
                    seed=model_seed,
                    )
        else:
            # For some unknown reason, using random_uniform on tensorflow to 
            # init one variable cannot work
            mb_init = np.random.uniform(
                    size=(instance_data_len, out_dim))
            mb_init = mb_init.astype(np.float32)
        std_dev = 1. / np.sqrt(out_dim/3)
        mb_init = mb_init * (2*std_dev) - std_dev
        memory_bank = tf.get_variable(
                'memory_bank', 
                initializer=mb_init,
                dtype=tf.float32,
                trainable=False,
                )
        if not use_tpu:
            label_init = tf.zeros_initializer
            all_label_kwarg = {
                    'shape':(instance_data_len),
                    }
            all_labels = tf.get_variable(
                    'all_labels',
                    initializer=label_init,
                    trainable=False,
                    dtype=tf.int64,
                    **all_label_kwarg
                    )

        if train:
            ### Randomly sample noise labels
            index_name = 'index'
            assert index_name in inputs, "Input should include index!"
            data_indx = inputs[index_name]
            noise_indx = tf.random_uniform(
                    shape=(batch_size, instance_k),
                    minval=0,
                    maxval=instance_data_len,
                    dtype=tf.int64)
            # data_memory [bs, out_dim]
            data_memory = tf.gather(memory_bank, data_indx, axis=0)
            noise_memory = tf.reshape(
                    tf.gather(memory_bank, noise_indx, axis=0),
                    [batch_size, instance_k, out_dim]
                    ) # [bs, k, out_dim]

            ### Compute the data distance and noise distance
            curr_out_ext = tf.expand_dims(curr_out, axis=1)
            data_dist = tf.reshape(
                    tf.matmul(
                        curr_out_ext, 
                        tf.expand_dims(data_memory, axis=2)),
                    [batch_size]) # [bs]
            noise_dist = tf.squeeze(
                    tf.matmul(
                        curr_out_ext, 
                        tf.transpose(noise_memory, [0,2,1])),
                    axis=1) # [bs, k]
            data_dist = tf.exp(data_dist / instance_t)
            noise_dist = tf.exp(noise_dist / instance_t)
            instance_Z = tf.constant(
                    2876934.2 / 1281167 * instance_data_len, 
                    dtype=tf.float32) 
            data_dist /= instance_Z
            noise_dist /= instance_Z
            add_outputs = [data_dist, noise_dist]
            
            ### Update the memory bank
            new_data_memory = data_memory*instance_m \
                    + (1-instance_m)*curr_out
            new_data_memory = tf.nn.l2_normalize(new_data_memory, axis=1)
            if not use_tpu:
                mb_update_op = tf.scatter_update(
                        memory_bank, 
                        data_indx, 
                        new_data_memory)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mb_update_op)
            else:
                # As scatter_update is not supported on TPU
                update_data_memory = new_data_memory - data_memory
                scatter_memory = tf.scatter_nd(
                        tf.expand_dims(data_indx, axis=1),
                        update_data_memory,
                        shape=memory_bank.shape)
                # On tpu, collecting all updates on each tpu core
                scatter_memory = \
                        tf.contrib.tpu.cross_replica_sum(scatter_memory)
                mb_update_op = tf.assign_add(
                        memory_bank, 
                        scatter_memory,
                        use_locking=False)
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mb_update_op)

            if not use_tpu:
                ### Update the labels
                lb_update_op = tf.scatter_update(
                        all_labels, 
                        data_indx, 
                        inputs['label'])
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, lb_update_op)
        else: # During validation
            all_dist = tf.matmul(curr_out, tf.transpose(memory_bank, [1, 0]))
            add_outputs = [
                    all_dist, 
                    tf.constant(
                        np.zeros([batch_size, instance_k]), 
                        dtype=tf.float32),
                    ]
        if not use_tpu:
            add_outputs.append(all_labels)
        else:
            add_outputs.append(inputs['label'])

        all_outputs.extend(add_outputs)
    return all_outputs
