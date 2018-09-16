# ============================================================================
# Author: John Martinez
# Email:  <john.r.martinez14@gmail.com>
# Date:   2018-09-01  15:44:17
# Brief:
# ============================================================================

import tensorflow as tf

from tensorflow.layers import batch_normalization as bn
from gans.ops.misc import attention
from gans.ops.conv import conv2d
from gans.ops.linear import dense
from gans.ops.conv import deconv2d
from gans.ops.norm import spectral_norm


def classifier(input_layer, is_train=True):
  """
  Args:
    input_layer:
    is_train:

  Returns:
  """
  batch_size = tf.shape(input_layer)[0]
  norm = spectral_norm

  with tf.variable_scope("cnn_classifier"):
    x = conv2d(
      input_layer,
      num_filters=64,
      filter_size=(2, 2),
      stride=(2, 2),
      pad="VALID",
      norm=norm,
      name="conv0")
    # x = attention(x, norm=norm, name="att0")
    x = tf.nn.leaky_relu(x)  # bn(x, training=is_train, name="bn0"))

    x = conv2d(
      x,
      num_filters=128,
      filter_size=(2, 2),
      stride=(2, 2),
      pad="VALID",
      norm=norm,
      name="conv1")
    # x = attention(x, norm=norm, name="att1")
    x = tf.nn.leaky_relu(x)  # bn(x, training=is_train, name="bn1"))

    x = conv2d(
      x,
      num_filters=256,
      filter_size=(2, 2),
      stride=(2, 2),
      pad="VALID",
      norm=norm,
      name="conv2")
    # x = attention(x, norm=norm, name="att2")
    x = tf.nn.leaky_relu(x)  # bn(x, training=is_train, name="bn2"))

    x = conv2d(
      x,
      num_filters=512,
      filter_size=(2, 2),
      stride=(2, 2),
      pad="VALID",
      norm=norm,
      name="conv3")
    # x = attention(x, norm=norm, name="att3")
    x = tf.nn.leaky_relu(x)  # bn(x, training=is_train, name="bn3"))

    x = tf.reshape(x, [batch_size, 2*2*512])
    x = tf.nn.dropout(x, 0.5)
    logits = dense(x, 10, norm=norm, name="logits")

    return tf.nn.softmax(logits), logits


def disc(input_layer, reuse=None, name="disc"):
  """
  Args:
    input_layer:
    name:
  
  Returns:
  """
  batch_size = tf.shape(input_layer)[0]
  norm = spectral_norm

  with tf.variable_scope(name, reuse=reuse):
    # 32 -> 16
    x = conv2d(
      input_layer,
      num_filters=64,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="conv0")
    x = tf.nn.leaky_relu(x)

    # 16 -> 8
    x = conv2d(
      x,
      num_filters=128,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="conv1")
    x = tf.nn.leaky_relu(x)

    # 8 -> 4
    x = conv2d(
      x,
      num_filters=256,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="conv2")
    x = tf.nn.leaky_relu(x)

    # 4 -> 2
    x = conv2d(
      x,
      num_filters=512,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="conv3")
    x = tf.nn.leaky_relu(x)

    x = tf.reshape(x, [batch_size, 2*2*512])
    x = dense(x, num_units=128, norm=norm, name="proj")
    logits = dense(x, num_units=1, norm=norm, name="logits")

    return tf.nn.sigmoid(logits), logits, x


def gen(z, reuse=None, name="gen"):
  """
  Args:
    z:
    name:
  
  Returns:
  """
  batch_size = tf.shape(z)[0]
  norm = spectral_norm

  with tf.variable_scope(name, reuse=reuse):
    # 128 -> 1024
    x = dense(z, 2*2*512, name="proj")

    # 1024 -> 2
    x = tf.reshape(x, [batch_size, 2, 2, 512])

    # 2 -> 4
    x = deconv2d(
      x,
      num_filters=512,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="deconv0")
    x = tf.nn.leaky_relu(x)  # bn(x, training=True), name="bn0")

    # 4 -> 8
    x = deconv2d(
      x,
      num_filters=256,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="deconv1")
    x = tf.nn.leaky_relu(x)  # bn(x, training=True), name="bn1")

    # 8 -> 16
    x = deconv2d(
      x,
      num_filters=128,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="deconv2")
    x = tf.nn.leaky_relu(x)  # bn(x, training=True), name="bn2")

    # 16 -> 32
    x = deconv2d(
      x,
      num_filters=64,
      filter_size=(4, 4),
      stride=(2, 2),
      pad="SAME",
      norm=norm,
      name="deconv3")
    x = tf.nn.leaky_relu(x)  # bn(x, training=True), name="bn3")

    x = conv2d(
      x,
      num_filters=3,
      # filter_size=(1, 1),
      norm=norm,
      name="conv0")

    return tf.nn.tanh(x)

