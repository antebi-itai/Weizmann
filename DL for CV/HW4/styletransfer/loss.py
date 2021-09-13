import torch
from torch import nn

##########################################################
# Gram Matrix
##########################################################

def gramMatrix(x):
  """
  This method computes the Gram Matrix of correlations between different
  channels. Given an input of shape BxCxHxW it collapses the spatial dimensions
  and computes the gram matric for each Cx(HxW) matrix in the batch.

  Args:
    x (Torch.Tensor) A tensor of images to extract.
    Has shape `(B, C, H, W)`. 
  Returns:            
    gram_matrices (Torch.Tensor) - The Gram matrices of x. 
    Gram matrix is computed over the last two dimensions - H and W.
    Has shape `(B, C, C)`. 
  """
  # BEGIN SOLUTION
  B, C, H, W = x.shape
  channels = x.reshape(B, C, H*W)
  gram_channels = channels @ channels.permute([0,2,1])
  return gram_channels
  # END SOLUTION

##########################################################
# Style Transfer Loss
##########################################################

def styleTransferLoss(content_lambdas, style_lambdas, 
                      input_contents, target_contents, 
                      input_styles, target_styles):
  """
  Args:
  content_lambdas (List(float)): Weights of content losses. Has length of
  number of content losses.
  style_lambdas (List(float)): Weights of style losses. Has length of
  number of style losses.
  input_contents (List(Torch.Tensor)) A list of the content features of the 
  optimized image.
  target_contents (List(Torch.Tensor)) A list of the content features of the 
  content image.
  input_styles (List(Torch.Tensor)) A list of the style features of the 
  optimized image.
  target_styles (List(Torch.Tensor)) A list of the style features of the 
  style image.
  Returns:            
    loss (Torch.Tensor) - The style transfer loss. 
  """
  # BEGIN SOLUTION
  # Compute losses
  content_len = len(content_lambdas)
  style_len = len(style_lambdas)

  content_losses = [(input_contents[i] - target_contents[i]).pow(2).sum() for i in range(content_len)]
  final_content_loss = sum([content_lambda*content_loss for content_lambda,content_loss in zip(content_lambdas, content_losses)])

  style_losses = [(gramMatrix(input_styles[i]) - gramMatrix(target_styles[i])).pow(2).sum() for i in range(style_len)]
  final_style_loss = sum([style_lambda*style_loss for style_lambda,style_loss in zip(style_lambdas, style_losses)])

  return final_content_loss + final_style_loss
  # END SOLUTION
