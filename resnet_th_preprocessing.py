from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import os, sys
import numpy as np

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  cropped_shape = control_flow_ops.with_dependencies(
      [rank_assertion],
      tf.stack([crop_height, crop_width, original_shape[2]]))

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  image = control_flow_ops.with_dependencies(
      [size_assertion],
      tf.slice(image, offsets, cropped_shape))
  return tf.reshape(image, cropped_shape)

def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs


def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(height * scale)
  new_width = tf.to_int32(width * scale)
  return new_height, new_width

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image


def _at_least_x_are_true(a, b, x):
  """At least `x` of `a` and `b` `Tensors` are true."""
  match = tf.equal(a, b)
  match = tf.cast(match, tf.int32)
  return tf.greater_equal(tf.reduce_sum(match), x)


def RandomSizedCrop_from_jpeg(
        image_str, 
        out_height, 
        out_width, 
        size_minval=0.08,
        ):
    shape = tf.image.extract_jpeg_shape(image_str)
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=0.1,
            aspect_ratio_range=(3. / 4, 4. / 3.),
            area_range=(size_minval, 1.0),
            max_attempts=100,
            use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, bbox = sample_distorted_bounding_box
    random_image = tf.image.decode_and_crop_jpeg(
            image_str, 
            tf.stack([bbox_begin[0], bbox_begin[1], \
                      bbox_size[0], bbox_size[1]]),
            channels=3)
    bad = _at_least_x_are_true(shape, tf.shape(random_image), 3)
    # central crop if bad
    min_size = tf.minimum(shape[0], shape[1])
    offset_height = tf.cast((shape[0] - min_size) / 2, tf.int32)
    offset_width = tf.cast((shape[1] - min_size) / 2, tf.int32)
    bad_image = tf.image.decode_and_crop_jpeg(
            image_str, 
            tf.stack([offset_height, offset_width, \
                      min_size, min_size]),
            channels=3)
    image = tf.cond(
            bad, 
            lambda: bad_image,
            lambda: random_image,
            )
    image = tf.cast(
            tf.image.resize_bicubic(
                [random_image], 
                [out_height, out_width])[0],
            dtype=tf.uint8)
    image.set_shape([out_height, out_width, 3])
    return image


def RandomBrightness(image, low, high):
    orig_dtype = image.dtype
    flt_image = tf.image.convert_image_dtype(image, tf.float32)
    rnd_bright = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    flt_image = flt_image*rnd_bright
    return tf.image.convert_image_dtype(flt_image, orig_dtype, saturate=True)


def RandomSaturation(image, low, high):
    orig_dtype = image.dtype
    flt_image = tf.image.convert_image_dtype(image, tf.float32)
    rnd_saturt = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    gry_image = tf.image.rgb_to_grayscale(flt_image)
    blend_image = flt_image*rnd_saturt + gry_image*(1-rnd_saturt)
    return tf.image.convert_image_dtype(blend_image, orig_dtype, saturate=True)


def RandomContrast(image, low, high):
    orig_dtype = image.dtype
    flt_image = tf.image.convert_image_dtype(image, tf.float32)
    rnd_contr = tf.random_uniform(
            shape=[], 
            minval=low, maxval=high, 
            dtype=tf.float32)
    mean_gray = tf.reduce_mean(tf.image.rgb_to_grayscale(flt_image))
    blend_image = flt_image*rnd_contr + mean_gray*(1-rnd_contr)
    return tf.image.convert_image_dtype(blend_image, orig_dtype, saturate=True)


def ColorJitter(image, seed_random=0, as_batch=False, shape_undefined=1):
    order_temp = tf.constant([0,1,2,3], dtype=tf.int32)
    order_rand = tf.random_shuffle(order_temp)

    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    fn_pred_fn_pairs = lambda x, image: [
            (tf.equal(x, order_temp[0]), 
                lambda :RandomSaturation(image, 0.6, 1.4)),
            (tf.equal(x, order_temp[1]), 
                lambda :RandomBrightness(image, 0.6, 1.4)),
            (tf.equal(x, order_temp[2]), 
                lambda :tf.image.random_hue(image, 0.4)),
            ]
    default_fn = lambda image: tf.image.random_contrast(image, 0.6, 1.4)
    #default_fn = lambda image: RandomContrast(image, 0.6, 1.4)

    def _color_jitter_one(_norm):
        orig_shape = tf.shape(_norm)
        for curr_idx in range(order_temp.get_shape().as_list()[0]):
            _norm = tf.case(
                    fn_pred_fn_pairs(order_rand[curr_idx], _norm), 
                    default=lambda : default_fn(_norm))
        if shape_undefined==0:
            _norm.set_shape(orig_shape)
        return _norm
    if as_batch:
        image = tf.map_fn(_color_jitter_one, image)
    else:
        image = _color_jitter_one(image)
    image = tf.image.convert_image_dtype(image, orig_dtype)
    return image


def ColorNormalize(image):
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - imagenet_mean) / imagenet_std

    return image


def ApplyGray(norm, prob_gray, as_batch=False):
    def _postprocess_gray(im):
        do_gray = tf.random_uniform(shape=[], minval=0, 
                                    maxval=1, dtype=tf.float32)
        def __gray(im):
            gray_im = tf.image.rgb_to_grayscale(im)
            gray_im = tf.tile(gray_im, [1,1,3])
            return gray_im
        return tf.cond(
                tf.less(do_gray, prob_gray), 
                lambda: __gray(im), 
                lambda: im)
    if as_batch:
        norm = tf.map_fn(_postprocess_gray, norm, dtype=norm.dtype)
    else:
        norm = _postprocess_gray(norm)
    return norm
