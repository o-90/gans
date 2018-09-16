# ==================================================================
# AUTHOR:
# EMAIL:
# DATE:
# BRIEF:
# ==================================================================

import numpy as np
import tensorflow as tf

from skimage.io import imsave


def _norm(x, eps=1e-12, name="norm"):
  """
  Args:
    x:
    eps:
    name:

  Returns:
    out:
  """
  # with tf.variable_scope(name):
  '''
    xs = tf.square(x)
    xs = tf.reduce_sum(xs)
    xs = tf.add(tf.sqrt(xs), tf.constant(eps))
    out = tf.div(x, xs)
    return out
  '''
  return x / (tf.reduce_sum(x ** 2) ** 0.5 + eps)


def _determine_transpose_dims(shape_or_tensor):
  """
  Args:
  Returns:
  """
  if isinstance(shape_or_tensor, tf.Tensor):
    shape = shape_or_tensor.get_shape().as_list()
  if isinstance(shape_or_tensor, list):
    shape = shape_or_tensor
  
  dim = len(shape)
  head = list(range(dim-2))
  tail = [dim-1, dim-2]
  return head + tail
  

def _determine_deconv_output_shape(input_length, filter_size, padding, stride):
  """
  Args:
  Returns:
  """
  input_length *= stride
  if padding == "VALID":
    input_length += max(filter_size - stride, 0)
  elif padding == "FULL":
    input_length -= (stride + filter_size - 2)
  return input_length


def _binary_argmax(t):
  """
  Args:
    t: tf.Tensor or numpy.array
  """
  t[t < 0.5] = 0
  t[t >= 0.5] = 1
  return t


def _save_div(x, y):
  """
  Args:
    x:
    y:

  Returns:
  """
  if y != 0.0:
    return x / y
  return x


def _frechet_dist(real, fake, eps=1e-10):
  """
  Frechet Distance

  |mu_r - mu_f|^2 - Trace(C_r + C_f - 2*(C_r * C_f)^(0.5))


  Args:
    real:
    fake:

  Returns:
    fid_score:
  """
  # get batch mean and covariance
  mu_r = tf.reduce_mean(real, axis=0)
  mu_f = tf.reduce_mean(fake, axis=0)

  # real covariance
  num_real_examples = tf.cast(tf.shape(real)[0], tf.float32)
  real_w = real - mu_r
  sigma_real = tf.matmul(real_w, real_w, transpose_a=True)
  sigma_real = sigma_real / (num_real_examples - 1)

  # fake covariance
  num_fake_examples = tf.cast(tf.shape(real)[0], tf.float32)
  fake_w = fake - mu_f
  sigma_fake = tf.matmul(fake_w, fake_w, transpose_a=True)
  sigma_fake = sigma_fake / (num_fake_examples - 1)

  # compute trace of the covariance matrices
  s, u, v = tf.svd(sigma_real)
  si = tf.where(tf.less(s, eps), s, tf.sqrt(s))
  sqrt_sigma_real = tf.matmul(tf.matmul(u, tf.diag(si)), v, transpose_b=True)
  sqrt_part = tf.matmul(
    sqrt_sigma_real, tf.matmul(sigma_fake, sqrt_sigma_real))
  trace = tf.trace(sigma_real + sigma_fake) - 2.0 * sqrt_part

  mean = tf.reduce_sum(tf.squared_difference(mu_r, mu_f))
  fid_score = trace + mean
  return fid_score


def _make_sample_images(images, epoch):
  """
  Args:
    images:

  Returns:
  """
  images = ((1.0 + images) * 127.5).astype(np.uint8)
  num_samples = images.shape[0]
  rows = int(np.sqrt(num_samples))
  while num_samples % rows != 0:
    rows -= 1
  nh, nw = rows, num_samples // rows
  h, w = images[0].shape[:2]
  arr = np.zeros((h * nh, w * nw, 3))
  for n, x in enumerate(images):
    j = n // nw
    i = n % nw
    arr[j*h:(j+1)*h, i*w:(i+1)*w] = x

  arr = arr.astype(np.uint8)
  imsave(f"sample_images_{epoch}.png", arr)
