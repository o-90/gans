# ============================================================================
# Author:  John Martinez
# Email:   <john.r.martinez14@gmail.com>
# Date:    2018-09-02  17:13:47
# Brief:
# ============================================================================

import tensorflow as tf


def attention(x, l=1.0, norm=None, name="att"):
  """
  Args:
  Returns:
  Notes:
  """
  with tf.variable_scope(name):
    orig_shp = x.get_shape().as_list()

    # make f, g, h
    f = conv2d(x, num_filters=orig_shp[-1], norm=norm, filter_size=(1, 1), pad="SAME", name="fconv1x1")
    g = conv2d(x, num_filters=orig_shp[-1], norm=norm, filter_size=(1, 1), pad="SAME", name="gconv1x1")
    h = conv2d(x, num_filters=orig_shp[-1], norm=norm, filter_size=(1, 1), pad="SAME", name="hconv1x1")

    f = tf.reshape(f, [-1, orig_shp[-1]])
    g = tf.reshape(g, [-1, orig_shp[-1]])
    h = tf.reshape(h, [-1, orig_shp[-1]])

    beta = tf.nn.softmax(tf.matmul(g, tf.transpose(f, [1, 0])))
    o = tf.matmul(beta, h)
    o = tf.reshape(o, [-1] + orig_shp[1:])
    out = l * o + x

    return out
