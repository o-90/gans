# ============================================================================
# AUTHOR: John Martinez
# EMAIL:  <john.r.martinez14@gmail.com>
# DATE:   2018-08-11  15:57:21
# BRIEF:  stuff
# ============================================================================

import os
import tensorflow as tf


class Data:
  """
  Args:
    config:
  """
  def __init__(self, config): 
    self.config = config

  def _parse(self):
    raise NotImplementedError

  def _process(self):
    raise NotImplementedError
    
  def get_data(self):
    raise NotImplementedError


class Cifar10Data(Data):
  def __init__(self, config):
    super().__init__(config)

  def _parse(self, ex):
    features = {
      "image": tf.FixedLenFeature([], tf.string),
      "label": tf.FixedLenFeature([], tf.int64),
    }
    parsed_ex = tf.parse_single_example(ex, features)

    # image
    image = tf.decode_raw(parsed_ex['image'], tf.uint8)
    image.set_shape((32 * 32 * 3))
    image = tf.reshape(image, [3, 32, 32])
    image = tf.transpose(image, [1, 2, 0])

    # label
    label = parsed_ex['label']

    return image, label

  def _process(self, image, label):
    # [0, 255] -> [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    # [0, 1] -> [-1, 1]
    image = tf.subtract(tf.constant(2.0) * image, tf.constant(1.0))
    # image = tf.image.per_image_standardization(image)

    # one-hot encode
    label = tf.eye(10)[label]

    return image, label

  def get_data(self, data="train"):
    path = os.path.join(self.config.data_dir, "cifar")

    if data == "train":
      path = os.path.join(path, "train.tfrecords")
    elif data == "valid":
      path = os.path.join(path, "validation.tfrecords")
    elif data == "predict":
      path = os.path.join(path, "test.tfrecords")
    else:
      raise ValueError(f"Not a valid data category, {data}.")

    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.map(self._parse)
    dataset = dataset.map(self._process)

    if data == "train":
      dataset = dataset.shuffle(self.config.buffer_size)

    dataset = dataset.repeat(self.config.num_epochs)
    dataset = dataset.batch(self.config.batch_size * self.config.num_devices)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels


class CelebAData(Data):
  def __init__(self, config):
    super().__init__(config)

  def _parse(self, ex):
    features = {
      "image": tf.FixedLenFeature([], tf.string)
    }
    parsed_ex = tf.parse_single_example(ex, features)

    image = tf.decode_raw(parsed_ex['image'], tf.uint8)
    image.set_shape((218 * 178 * 3))
    image = tf.reshape(image, [218, 178, 3])
    return image

  def _process(self, image):
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.subtract(tf.constant(2.0) * image, tf.constant(1.0))
    return image

  def get_data(self, data="train"):
    path = os.path.join(self.config.data_dir, "celeba")

    if data == "train":
      path = os.path.join(path, "train.tfrecords")
    elif data == "valid":
      path = os.path.join(path, "valid.tfrecords")
    else:
      raise ValueError(f"Not a valid data category, {data}.")

    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.map(self._parse)
    dataset = dataset.map(self._process)

    if data == "train":
      dataset = dataset.shuffle(self.config.buffer_size)

    dataset = dataset.repeat(self.config.num_epochs)
    dataset = dataset.batch(self.config.batch_size * self.config.num_devices)
    iterator = dataset.make_one_shot_iterator()
    images = iterator.get_next()

    return images
