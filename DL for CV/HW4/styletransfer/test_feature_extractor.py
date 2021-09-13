import unittest
import torch
from torchvision import transforms
from PIL import Image
from feature_extractor import FeatureExtractor
from utils import input_transforms

##########################################################
# Feature Extractor
##########################################################

class TestFeatureExtractor(unittest.TestCase):
  def setUp(self):
    self.extractor = FeatureExtractor()
    self.image_path = 'data/cat.jpg'
    self.image = Image.open(self.image_path)
    self.image_input = input_transforms(self.image)[None, ...]
    self.layers_dict = {
      FeatureExtractor.CONTENT_KEY: [22],
      FeatureExtractor.STYLE_KEY: [1, 6, 11, 20, 29]
    }

  def testExtract(self):
    output_dict = self.extractor.extract(self.image_input, **self.layers_dict)   
    for key in FeatureExtractor.KEY_LIST:
      self.assertTrue(len(self.layers_dict[key]) == len(output_dict[key]), msg=f"Did non get the expected amount of outputs for {key}")
