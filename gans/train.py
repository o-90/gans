#!/usr/bin/env python 

# ============================================================================
# AUTHOR:   John Martinez
# EMAIL:    <john.r.martinez14@gmail.com>
# DATE:     2018-09-15  12:40:43
# BRIEF:    
# ============================================================================

import os
import numpy as np
import tensorflow as tf

from gans import utils
from argparse import ArgumentParser
from gans.data import Cifar10Data
from datetime import datetime
from gans.config import Config
from gans.networks.cifar10.network import disc
from gans.networks.cifar10.network import gen


class Train:
  def __init__(self, config, data):
    self._data = data
    self.config = config
    self.graph = tf.Graph()
    self.data = self._data(self.config)

  #TODO(gobrewers14): move to utils
  @staticmethod
  def _avg_grads(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
      grads = [g for g, _ in grad_and_vars]
      grad = tf.reduce_mean(grads, 0)

      v = grad_and_vars[0][1]
      grad_and_var = (grad, v)
      average_grads.append(grad_and_var)
    return average_grads

  def _build(self):
    with self.graph.as_default():
      with tf.device("/cpu:0"):
        # images
        train_images, _ = self.data.get_data()
        train_images = tf.split(
          train_images,
          num_or_size_splits=self.config.num_devices,
          axis=0)
        
        # latent space
        z = tf.random_normal(shape=[self.config.batch_size, 128], name="z")

      d_opt = tf.train.AdamOptimizer(
        learning_rate=self.config.d_lr, beta1=self.config.beta1, beta2=self.config.beta2)
      g_opt = tf.train.AdamOptimizer(
        learning_rate=self.config.g_lr, beta1=self.config.beta1, beta2=self.config.beta2)

      d_grads = []
      g_grads = []

      self.d_losses = []
      self.g_losses = []

      reuse_vars = None

      for idx, (train_imgs, device) in enumerate(zip(train_images, self.config.devices)):
        with tf.device(f"{device}"):
          with tf.name_scope(f"device_{idx}"):
            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_vars):

              Gz = gen(z=z)
      
              _, Dx, _ = disc(input_layer=train_imgs)
              _, D_Gz, _ = disc(input_layer=Gz, reuse=True)
      
              # d_loss = tf.reduce_mean(tf.nn.softplus(-Dx) + tf.nn.softplus(D_Gz))
              # g_loss = tf.reduce_mean(tf.nn.softplus(-D_Gz))
      
              d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=Dx,
                labels=0.9 * tf.ones_like(Dx))
              d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_Gz,
                labels=tf.zeros_like(D_Gz))
            
              d_loss = tf.reduce_mean(d_loss_real + d_loss_fake)
            
              g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=D_Gz,
                labels=0.9 * tf.ones_like(D_Gz))
              g_loss = tf.reduce_mean(g_loss)
  
              if idx == 0:
                d_vars = [v for v in tf.trainable_variables() if v.name.startswith("disc")]
                g_vars = [v for v in tf.trainable_variables() if v.name.startswith("gen")]
      
              d_grad = d_opt.compute_gradients(d_loss, var_list=d_vars)
              g_grad = g_opt.compute_gradients(g_loss, var_list=g_vars)

              d_grads.append(d_grad)
              g_grads.append(g_grad)

              self.d_losses.append(d_loss)
              self.g_losses.append(g_loss)

              reuse_vars = True

      # average gradients
      avg_d_grads = self._avg_grads(d_grads)
      avg_g_grads = self._avg_grads(g_grads)

      # create train op
      self.d_train_op = d_opt.apply_gradients(avg_d_grads)
      self.g_train_op = g_opt.apply_gradients(avg_g_grads)

      # initialize all variables
      self.init_op = tf.global_variables_initializer()

    '''# --- validation ---
    valid_images, _ = data.get_data(data="valid")
    valid_z = tf.random_normal([c.batch_size, 128])

    # gen for fid score
    Gz_valid = gen(z=valid_z, reuse=True)
    
    # disc
    real_prob_op, _, real_activ = disc(input_layer=valid_images, reuse=True)
    fake_prob_op, _, fake_activ = disc(input_layer=Gz_valid, reuse=True)

    fid_op = utils._frechet_dist(real_activ, fake_activ)'''

  def train(self):
    self._build()
    with tf.Session(graph=self.graph) as sess:
      sess.run(self.init_op)
      
      d_train_losses = []
      g_train_losses = []

      start_time = datetime.now()
      print("start training...")
      print('-' * 79)
      for i in range(self.config.num_epochs):
        _ = sess.run(self.d_train_op)
        _ = sess.run(self.g_train_op)

        d_train_loss, g_train_loss = sess.run([self.d_losses, self.g_losses])
        d_train_losses.append(d_train_loss)
        g_train_losses.append(g_train_loss)


        if i % 1000 == 0:
          update_str = f"update: {i}\n"
          update_str += f"\td_loss:\t{np.mean(d_train_losses):.3f} ± {np.std(d_train_losses):.3f}\n"
          update_str += f"\tg_loss:\t{np.mean(g_train_losses):.3f} ± {np.std(g_train_losses):.3f}\n"
          update_str += f"\ttime:\t{datetime.now()-start_time}\n"
          update_str += "-" * 79
          print(update_str)


def parse_args(parser):
  parser.add_argument("--batch-size", type=int, dest="batch_size", default=32)
  parser.add_argument("--num-epochs", type=int, dest="num_epochs", default=100000)
  parser.add_argument("--devices", type=list, dest="devices", default=["/gpu:0", "/gpu:1"])
  options = parser.parse_args()
  return options


def main(options):
  params = {
    "batch_size": options.batch_size,
    "num_epochs": options.num_epochs,
    "devices": options.devices,
  }

  config = Config(**params)
  trainer = Train(config=config, data=Cifar10Data)
  trainer.train()


if __name__ == "__main__":
  parser = ArgumentParser(description="")
  options = parse_args(parser)
  main(options)
