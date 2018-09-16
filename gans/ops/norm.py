# ============================================================================
# Author:  John Martinez
# Email:   <john.r.martinez14@gmail.com>
# Date:    2018-09-02  17:13:47
# Brief:
# ============================================================================

import tensorflow as tf


def spectral_norm(weight, num_iters=3, name="sn"):
  """
  Args:
    weight: a tf.Tensor representing the weights to normalize
    num_iters: the number of spectral power iteration to perform
    name: the name of the op

  Returns:
    w_norm: the normalized weight tensor

  Notes:
    Spectral normalization is described in
    `Spectral Normalization for Generative Adversarial Networks`
    https://arxiv.org/pdf/1802.05957.pdf
  """
  with tf.variable_scope(name):
    shp = weight.get_shape().as_list()
    weight = tf.reshape(weight, [-1, shp[-1]])
    new_shp = weight.get_shape().as_list()

    u = tf.get_variable(
      name="u",
      dtype=tf.float32,
      shape=[1, shp[-1]],
      trainable=False,
      initializer=tf.truncated_normal_initializer())

    u_hat = u
    v_hat = None

    def _power_iter(i, ui, vi, w):
      v_ = tf.matmul(ui, tf.transpose(w))
      v_hat = tf.nn.l2_normalize(v_)

      u_ = tf.matmul(v_hat, w)
      u_hat = tf.nn.l2_normalize(u_)

      return tf.add(i, 1), u_hat, v_hat, w

    _, u_hat, v_hat, _ = tf.while_loop(
      cond=lambda i, *_: i < num_iters,
      body=_power_iter,
      loop_vars=(
        tf.constant(0),
        u,
        tf.zeros([1] + new_shp[:-1]),
        weight))

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, weight), tf.transpose(u_hat))
    with tf.control_dependencies([u.assign(u_hat)]):
      w_norm = weight / sigma
      w_norm = tf.reshape(w_norm, shp)

    return w_norm
