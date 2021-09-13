from pathlib import Path
from PIL import Image
import pandas as pd
import utils
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils
from feature_extractor import FeatureExtractor
from loss import styleTransferLoss

##########################################################
# Experiment
##########################################################
class Experiment:
  """Facilitates creating a style transferred image. You should use the
  components you implemented in previous clauses in the exercise and find the 
  best hyperparameters."""

  def __init__(self, content_paths, style_paths, config):
    """
    Args:
      content_paths (List(pathlib.Path)): Path to the content images.
      style_paths (List(pathlib.Path)): Path to the style images.
      config (dict): Configuration dictionary. 
      Contains parameters from training & evaluation.
    """    
    # BEGIN SOLUTION
    self.content_paths = content_paths
    self.style_paths = style_paths
    for key, value in config.items():
      setattr(self, key, value)

    self.content_extractor = FeatureExtractor(device=self.device, model_type=self.model_type)
    self.style_extractor = FeatureExtractor(device=self.device, model_type=self.model_type)
    self.input_extractor = FeatureExtractor(device=self.device, model_type=self.model_type)
    # END SOLUTION

  def optimize(self, content_path, style_path):
    """ Style transfer a specific image. Should contain image loading,
    definition of the optimization objective and optimization loop.
    Args:
      content_path (pathlib.Path): Path to the content image.
      style_path (pathlib.Path): Path to the style image.

    Returns:
      st_image (torch.Tensor): The style-transferred image.
    """
    # BEGIN SOLUTION
    with Image.open(content_path) as content_image:
      with Image.open(style_path) as style_image:
        # load images
        C = utils.input_transforms(content_image).unsqueeze(0).to(self.device)
        S = utils.input_transforms(style_image).unsqueeze(0).to(self.device)
        if self.from_noise:
          I = torch.randn_like(C, requires_grad=True).to(self.device)
        else:
          I = C.clone().requires_grad_(True)
        
        optimizer = torch.optim.Adam([I], lr=self.lr)
        for epoch in tqdm(range(self.epochs)):
          # extract features
          target_contents = self.content_extractor.extract(C, content=self.content_layers)['content']
          target_styles = self.style_extractor.extract(S, style=self.style_layers)['style']
          input_dict = self.input_extractor.extract(I, content=self.content_layers, style=self.style_layers)
          input_contents, input_styles = input_dict['content'], input_dict['style']
          # calculate the loss
          loss = styleTransferLoss(self.content_lambdas, self.style_lambdas, 
                      input_contents, target_contents, 
                      input_styles, target_styles)
          # train
          self.input_extractor.model.zero_grad()
          I.grad = None
          loss.backward()
          optimizer.step()
    return I
    # END SOLUTION

  def run(self):
    """ Run an entire experiment. Iterates through pairs of content-style images
    and creates their style-transfer result and saves them.
    """
    # train and evaluate every image
    for content_path, style_path in tqdm(zip(self.content_paths, self.style_paths)):
      # create style-transfered image:
      st_image = self.optimize(content_path, style_path)
      # save image
      st_pil = utils.to_pil(st_image[0])
      st_pil.save(utils.ROOT / f'{Path(content_path).stem}_{Path(style_path).stem}.png')
