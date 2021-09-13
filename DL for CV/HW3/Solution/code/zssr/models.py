from torch import nn
import utils
from functools import partial

##########################################################
# Basic Model
##########################################################
class ZSSRNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    super().__init__()
    self.scale_factor = scale_factor
    
    self.first_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=kernel_size, padding=kernel_size//2)
    self.first_relu = nn.ReLU()
    self.base_conv = nn.ModuleList([nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=kernel_size, padding=kernel_size//2),
                                                  nn.ReLU())
                                    for i in range(6)])
    self.last_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=kernel_size, padding=kernel_size//2)
    #END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """    
    # BEGIN SOLUTION
    resized_x = utils.rr_resize(x, self.scale_factor)
    
    out = self.first_relu(self.first_conv(resized_x))
    for conv_layer in self.base_conv:
      out = conv_layer(out)
    out = self.last_conv(out)
    return out
    # END SOLUTION


##########################################################
# Advanced Model
##########################################################
class ZSSRResNet(nn.Module):
  """A super resolution model. """

  def __init__(self, scale_factor, kernel_size=3):
    """ Trains a ZSSR model on a specific image.
    Args:
      scale_factor (int): ratio between SR and LR image sizes.
      kernel_size (int): size of kernels to use in convolutions.
    """
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION

  def forward(self, x):
    """ Apply super resolution on an image.
    First, resize the input image using `utils.rr_resize`.
    Then pass the image through your CNN.
    Finally, add the CNN's output in a residual manner to the original resized
    image.
    Args:
      x (torch.Tensor): LR input.
      Has shape `(batch_size, num_channels, height, width)`.

    Returns:
      output (torch.Tensor): HR input.
      Has shape `(batch_size, num_channels, self.s * height, self.s * width)`.
    """   
    # BEGIN SOLUTION
    raise NotImplementedError
    # END SOLUTION


##########################################################
# Original Model
##########################################################
class ZSSROriginalNet(nn.Module):
  """A super resolution model. """

  def __init__(self, **kwargs):
    # BEGIN SOLUTION
    raise NotImplementedError     
    # END SOLUTION

  def forward(self, x):
    # BEGIN SOLUTION
    raise NotImplementedError     
    # BEGIN SOLUTION
