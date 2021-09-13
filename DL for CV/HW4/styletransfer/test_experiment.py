import unittest
import torch
from torchvision import transforms
from pathlib import Path
from experiment import Experiment
import utils

##########################################################
# Experiment
##########################################################

class TestExperiment(unittest.TestCase):
  def setUp(self):
    self.config = {
      'model_type': 'vgg19',
      'content_layers': [22],
      'content_lambdas': [1e0],
      'style_layers': [1, 6, 11, 20, 29],
      'style_lambdas': [1e3/n**2 for n in [64, 128, 256, 512, 512]],  
      'device': 'cuda' if torch.cuda.is_available() else 'cpu',
      'verbose': True,
      'lr': 0.0001, # learning rate
      'epochs': 5, # number of epochs to run
      'show_interval': 1, # plot after this number of epochs
      'from_noise': False
    }
    self.content_path = 'data/cat.jpg'
    self.style_path = 'data/kadishman.jpg'

    self.experiment = Experiment([self.content_path], [self.style_path], self.config)

  def testOptimize(self):
    self.experiment.optimize(self.content_path, self.style_path)
    self.assertTrue(True)
