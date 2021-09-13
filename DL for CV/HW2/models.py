import torch  # noqa

from nn import Module, Linear  # noqa
from functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']


#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()
    # BEGIN SOLUTION
    self.linear = Linear(in_dim, num_classes)
    self._modules = ['linear']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    y = self.linear(x, ctx=ctx)
    return y
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes, hidden_size):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()
    # BEGIN SOLUTION
    self.linear1 = Linear(in_dim, hidden_size)
    self.linear2 = Linear(hidden_size, hidden_size)
    self.linear3 = Linear(hidden_size, num_classes)
    self._modules = ['linear1', 'linear2', 'linear3']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Note: `cross_entropy_loss` expects predictions BEFORE applying `softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    h1 = relu(self.linear1(x, ctx=ctx), ctx=ctx)
    h2 = relu(self.linear2(h1, ctx=ctx), ctx=ctx)
    y = self.linear3(h2, ctx=ctx)
    return y
    # END SOLUTION
