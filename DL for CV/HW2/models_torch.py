import torch  # noqa

from torch.nn import Module, Linear  # noqa
from torch.nn.functional import relu  # noqa

__all__ = ['SoftmaxClassifier', 'MLP']


#################################################
# SoftmaxClassifier
#################################################

class SoftmaxClassifier(Module):
  """A simple softmax classifier"""

  def __init__(self, in_dim, num_classes):
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION
    self.linear = Linear(in_dim, num_classes)
    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    return self.linear(x)
    # END SOLUTION


#################################################
# MLP
#################################################

class MLP(Module):
  """A multi-layer perceptron"""

  def __init__(self, in_dim, num_classes, hidden_size):  # YOU CAN MODIFY THIS LINE AND ADD ARGUMENTS
    super().__init__()  # This line is important in torch Modules.
                        # It replaces the manual registration of parameters
                        # and sub-modules (to some extent).
    # BEGIN SOLUTION
    self.linear1 = Linear(in_dim, hidden_size)
    self.linear2 = Linear(hidden_size, hidden_size)
    self.linear3 = Linear(hidden_size, num_classes)
    # END SOLUTION

  def forward(self, x):
    """Computes the forward function of the network.

    Note: `F.cross_entropy` expects predictions BEFORE applying `F.softmax`.

    Args:
      x (torch.Tensor): The input tensor, has shape of `(batch_size, in_dim)`.

    Returns:
      y (torch.Tensor): The output tensor, has shape of `(batch_size, num_classes)`.
    """
    # BEGIN SOLUTION
    h1 = relu(self.linear1(x))
    h2 = relu(self.linear2(h1))
    y = self.linear3(h2)
    return y
    # END SOLUTION
