# ============================================================================
# AUTHOR: John Martinez
# EMAIL:  <john.martinez14@gmail.com>
# DATE:   2018-08-11  13:50:45
# BRIEF:
# ============================================================================

import os
import platform


class Config:
  def __init__(self,
               home_dir=None,
               working_dir=None,
               data_dir="/data/working/vagrant/data",
               meta_dir="/data/working/vagrant/metadata",
               model_dir="/data/working/vagrant/models",
               batch_size=64,
               num_epochs=100000,
               num_devices=1,
               buffer_size=1024,
               beta1=0.0,
               beta2=0.9,
               n_disc=3,
               d_lr=2e-4,
               g_lr=1e-4):

    # dir structure
    self.home_dir = home_dir or os.environ.get('HOME')
    self.working_dir = working_dir or os.path.join(self.home_dir, "workspace")
    self.data_dir = data_dir
    self.meta_dir = meta_dir
    self.model_dir = model_dir

    # experiment
    self.platform = platform.platform()
    self.experiment_id = self.platform  # + plus time or whatever

    # tensorflow constants
    self.batch_size = batch_size
    self.num_epochs = num_epochs
    self.num_devices = num_devices
    self.buffer_size = buffer_size
    self.d_lr = d_lr
    self.g_lr = g_lr

    # self.seed
    self.beta1 = beta1
    self.beta2 = beta2
    self.n_disc = n_disc
