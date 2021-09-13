import torch  # noqa

from nn import Module, Linear, Conv2d, MaxPool2d  # noqa
from functional import relu, view  # noqa

__all__ = ['ConvNet']


#################################################
# ConvNet
#################################################

class ConvNet(Module):
  """A deep convolutional neural network"""

  def __init__(self, in_channels, num_classes):
    super().__init__()
    # BEGIN SOLUTION
    self.conv1 = Conv2d(in_channels, 16, kernel_size=3, padding=1)
    self.conv2 = Conv2d(16, 16, kernel_size=3, padding=1)
    self.max1 = MaxPool2d(kernel_size=2, stride=2)
    self.conv3 = Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv4 = Conv2d(32, 32, kernel_size=3, padding=1)
    self.max2 = MaxPool2d(kernel_size=2, stride=2)
    self.conv5 = Conv2d(32, 32, kernel_size=3, padding=1)
    self.linear = Linear(2048, num_classes)

    self._modules = ['conv1', 'conv2', 'max1', 'conv3', 'conv4', 'max2', 'conv5', 'linear']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_channels, height, width)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    out = x
    out = relu(self.conv1(out, ctx=ctx), ctx=ctx)
    out = relu(self.conv2(out, ctx=ctx), ctx=ctx)
    out = self.max1(out, ctx=ctx)
    out = relu(self.conv3(out, ctx=ctx), ctx=ctx)
    out = relu(self.conv4(out, ctx=ctx), ctx=ctx)
    out = self.max2(out, ctx=ctx)
    out = self.conv5(out, ctx=ctx)
    out = view(out, (-1, 2048), ctx=ctx)
    out = self.linear(out, ctx=ctx)
    return out
    # END SOLUTION

