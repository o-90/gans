# ============================================================================
# Author:  John Martinez
# Email:   <john.r.martinez14@gmail.com>
# Date:    2018-09-02  17:13:47
# Brief:
# ============================================================================

import tensorflow as tf


def dense(x, num_units, bias=True, update=None, reuse=None, norm=None,
          name="dense"):
  """
  Args:
    x:
    num_units:
    bias:
    update:
    reuse:
    name:

  Returns:
    out:
  """
  # incoming shape
  shp = x.get_shape().as_list()

  with tf.variable_scope(name, reuse=reuse):
    dim = len(shp)
    w = tf.get_variable(
      name='w',
      dtype=tf.float32,
      shape=[shp[-1], num_units],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

    if norm:
      out = tf.matmul(x, norm(w))

    else:
      out = tf.matmul(x, w)

    if bias:
      b = tf.get_variable(
        name='b',
        dtype=tf.float32,
        shape=[num_units],
        initializer=tf.constant_initializer(0.0))

      out = tf.nn.bias_add(out, b)
    return out
