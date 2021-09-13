import unittest
import torch
from torchvision import transforms
from PIL import Image
from loss import gramMatrix, styleTransferLoss
from utils import input_transforms

##########################################################
# Gram Matrix
##########################################################

class TestGramMatrix(unittest.TestCase):
 
  def testGramMatrix(self):
    B, C, H, W = 2, 10, 14, 14
    input_x = torch.randn(B, C, H, W)
    output_x = gramMatrix(input_x)   
    self.assertTrue(output_x.shape == (B, C, C), msg=f"Output shape is incompatible. Expected ({B},{C},{C}) got {output_x.shape}")

##########################################################
# Style Transfer Loss
##########################################################

class TestStyleTransferLoss(unittest.TestCase):
  def setUp(self):
    pass
  
  def testLossSame(self):
    num_contents, num_styles = 1, 5
    B, C, H, W = 1, 16, 56, 56
    content_lambdas = [1] * num_contents
    style_lambdas = [1] * num_styles
    input_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    #target_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    target_contents = [content.clone().detach() for content in input_contents]
    input_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    #target_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    target_styles = [style.clone().detach() for style in input_styles]
    loss = styleTransferLoss(content_lambdas, style_lambdas, input_contents, target_contents, input_styles, target_styles)
    self.assertTrue(loss == 0, msg=f"Loss should be zero for identical input and target.")
  
  def testLossOnlyContent(self):
    num_contents, num_styles = 5, 0
    B, C, H, W = 1, 16, 56, 56
    content_lambdas = [1] * num_contents
    style_lambdas = [1] * num_styles
    input_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    target_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    input_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    target_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    loss = styleTransferLoss(content_lambdas, style_lambdas, input_contents, target_contents, input_styles, target_styles)
    self.assertTrue(loss > 0, msg=f"Loss should be positive.")
  
  def testLossOnlyStyle(self):
    num_contents, num_styles = 0, 5
    B, C, H, W = 1, 16, 56, 56
    content_lambdas = [1] * num_contents
    style_lambdas = [1] * num_styles
    input_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    target_contents = [torch.randn(B, C, H, W) for i in range(num_contents)]
    input_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    target_styles = [torch.randn(B, C, H, W) for i in range(num_styles)]
    loss = styleTransferLoss(content_lambdas, style_lambdas, input_contents, target_contents, input_styles, target_styles)
    self.assertTrue(loss > 0, msg=f"Loss should be positive.")
