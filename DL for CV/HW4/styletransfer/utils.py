import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from livelossplot import PlotLosses  
from pathlib import Path


#################################################
# PROVIDED: Constants
#################################################
ROOT = Path("/content/hw4/styletransfer")
DATA_ROOT = ROOT / 'data'


##########################################################
# PROVIDED: UnNormalize Transform
##########################################################

class UnNormalize:
    """
    In courtesy of https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    Usage:
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    unorm(tensor)
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

class Clip:
  """Clips the values of an tensor into the requested range."""
  def __init__(self, min, max):
    self.min = min
    self.max = max
  
  def __call__(self, tensor):
    return torch.clip(tensor, self.min, self.max)


##########################################################
# PROVIDED: Transforms
##########################################################

input_transforms = transforms.Compose([
    transforms.Resize(512),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225])
])

vis_transforms = transforms.Compose([
    UnNormalize(mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]),
    Clip(min=0, max=1),
    transforms.ToPILImage()
])


##########################################################
# PROVIDED: plot_image
##########################################################

def to_pil(image_input):
  """
  Args:
      image_input (Tensor): Tensor image of size (C, H, W) that has undergone
      input_transforms transformations.
  Returns:
      pil: PIL image.
  """
  pil_image = vis_transforms(image_input.clone().detach().cpu())
  return pil_image

##########################################################
# PROVIDED: Visualizer
##########################################################
class Visualizer:
  """Visualization using the liveplotloss library."""
  def __init__(self):
    self.liveloss = PlotLosses()

  def update(self, train_loss):
    """
    Args:
      train_loss (float): current loss in the training process.
    """
    train_epoch_loss = train_loss  
    logs = {}
    logs[f'loss'] = train_epoch_loss
    self.liveloss.update(logs)
    self.liveloss.send()