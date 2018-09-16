# ============================================================================
# Author:  John Martinez
# Email:   <john.r.martinez14@gmail.com>
# Date:    2018-09-02  17:13:47
# Brief:
# ============================================================================

import tensorflow as tf
from gans import utils


def conv2d(x,
           num_filters,
           filter_size=(3, 3),
           stride=(1, 1),
           pad="SAME",
           bias=True,
           update=None,
           reuse=None,
           norm=None,
           name="conv2d"):
  """
  Args:
    x:
    num_filters:
    filter_size:
    stride:
    pad:
    bias:
    reuse:
    name:

  Returns:
    out:
  """
  shp = x.get_shape().as_list()
  _, _, _, channel = shp
  with tf.variable_scope(name, reuse=reuse):
    w = tf.get_variable(
      name='w',
      dtype=tf.float32,
      shape=[filter_size[0], filter_size[1], channel, num_filters],
      initializer=tf.truncated_normal_initializer(stddev=0.02))
    if norm:
      out = tf.nn.conv2d(
        input=x,
        filter=norm(w),
        strides=[1, stride[0], stride[1], 1],
        padding=pad)

    else:
      out = tf.nn.conv2d(
        input=x,
        filter=w,
        strides=[1, stride[0], stride[1], 1],
        padding=pad)

    if bias:
      b = tf.get_variable(
        name='b',
        dtype=tf.float32,
        shape=[num_filters],
        initializer=tf.constant_initializer(0.0))

      out = tf.nn.bias_add(out, b)
    return out


def deconv2d(x, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME",
             bias=True, norm=None, reuse=None, name="deconv"):
  """
  Args:
  Returns:
  """
  batch_size, height, width, channel = x.get_shape().as_list()
  with tf.variable_scope(name):
    out_dim_h = utils._determine_deconv_output_shape(
      height, filter_size[0], padding=pad, stride=stride[0])
    out_dim_w = utils._determine_deconv_output_shape(
      width, filter_size[1], padding=pad, stride=stride[1])

    w = tf.get_variable(
      name='w',
      dtype=tf.float32,
      shape=[filter_size[0], filter_size[1], num_filters, channel],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    if norm:
      out = tf.nn.conv2d_transpose(
        value=x,
        filter=norm(w),
        output_shape=[batch_size, out_dim_h, out_dim_w, num_filters],
        strides=[1, stride[0], stride[1], 1],
        padding=pad)

    else:
      out = tf.nn.conv2d_transpose(
        value=x,
        filter=w,
        output_shape=[batch_size, out_dim_h, out_dim_w, num_filters],
        strides=[1, stride[0], stride[1], 1],
        padding=pad)

    if bias:
      b = tf.get_variable(
        name='b',
        dtype=tf.float32,
        shape=[num_filters],
        initializer=tf.constant_initializer(0.0))

      out = tf.nn.bias_add(out, b)
    return out
