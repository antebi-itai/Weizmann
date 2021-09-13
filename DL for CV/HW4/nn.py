import math  # noqa

import torch  # noqa

from hw2_nn import Module
from hw3_nn import Linear  # noqa
from hw3_nn import *  # noqa
from functional import add, mul, linear  # noqa
from functional import sigmoid, tanh  # noqa
from functional import cat, unbind, embedding  # noqa
from functional import *  # noqa
from hw3_nn import __all__ as __old_all__

__new_all__ = ['LSTMCell', 'Embedding']
__all__ = __old_all__ + __new_all__

class Embedding(Module):
  """ Map each the entry of the input to a learnable embedding. """
  def __init__(self, vocab_size, embedding_dim):
    """Creates an embedding layer.

    In this method you should:
      * Create a weight parameter (call it `weight`).
      * Add these parameter names to `self._parameters`.
      * Initialize the parameters
      
    Args:
      vocab_size (int): The size of the vocabulary
      embedding_dim (int): The dimension of the embedding vectors
    """
    super().__init__()
    # BEGIN SOLUTION
    self.weight = torch.empty(vocab_size, embedding_dim)
    self._parameters += ['weight']
    self.init_parameters()
    # END SOLUTION

  def init_parameters(self):
    """Initializes the parameters of the Embedding layer.
    You should initialize the weight with normal distribution (Zero mean and 0.01 STD). 
    """
    # BEGIN SOLUTION
    self.weight.normal_(mean=0, std=0.01)
    # END SOLUTION

  def forward(self, x, ctx=None):
    """ Computes the embedding of that input.
    
    Formula:
      for each entry of x
        return W[x[i][j], :]

      Args:
      x (torch.LongTensor): The input tensor. Has shape `(batch_size, sequence_length)`.
        x entries must be b non-negative and smaller than the vocabulary size.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      y (torch.Tensor): The embedding tensor, Has shape `(batch_size, sequence_length, embedding_size)`.
      """
    # BEGIN SOLUTION
    return embedding(x, self.weight, ctx=ctx)
    # END SOLUTION
    
class LSTMCell(Module):
  """LSTM cell, given an input vector, an hidden state and a cell state reutrn
  the next hidden state and cell state. 
  For the formulas see https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
  """
  def __init__(self, in_dim, hidden_dim):
    """Creates a LSTM cell.

    In this method you should:
      * Store the parameters (in_dim, hidden_dim) as class variables
      * Create linear layers (Using `cat` you can create fewer Linear layer)
      * Store the layers in `self._modules` (Refer to previous HW for explanation)

    Args:
      in_dim (int): The size of the input feature vector
      hidden_dim (int): The size of the hidden feature vectors
    """
    super().__init__()
    # BEGIN SOLUTION
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.linear = Linear( in_dim = self.in_dim + self.hidden_dim, 
                          out_dim = 4 * self.hidden_dim)
    self._modules += ['linear']
    # END SOLUTION

  def forward(self, xt, ht_1=None, ct_1=None, ctx=None):
    """Computes the the next hidden and cell states using the LSTM formula.

    Args:
      xt (torch.Tensor): The input tensor. Has shape `(batch_size, in_dim)`.
      ht_1 (torch.Tensor): The previous hidden state tensor. Has shape `(batch_size, hidden_dim)`.
      ct_1 (torch.Tensor): The previous cell state tensor. Has shape `(batch_size, hidden_dim)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Note: if ht_1 or ct_1 are None you should initialize them with zeros.

    Returns:
      ht (torch.Tensor): The hidden state tensor. Has shape `(batch_size, hidden_dim)`.
      ct (torch.Tensor): The cell state tensor. Has shape `(batch_size, hidden_dim)`.
    """
    assert xt.dim() == 2, f"xt should be a batch of vectors (2D). got: {xt.dims()}"
    # BEGIN SOLUTION
    batch_size, _ = xt.shape
    if ht_1 is None: 
      ht_1 = torch.zeros(batch_size, self.hidden_dim, dtype=xt.dtype, device=xt.device)
    if ct_1 is None: 
      ct_1 = torch.zeros(batch_size, self.hidden_dim, dtype=xt.dtype, device=xt.device)
    
    concat_input = cat([ht_1, xt], dim=1, ctx=ctx)
    product = view(self.linear(concat_input, ctx=ctx), (batch_size, 4, -1), ctx=ctx)
    splitted_product = unbind(product, dim=1, ctx=ctx)

    i = sigmoid(splitted_product[0], ctx=ctx)
    f = sigmoid(splitted_product[1], ctx=ctx)
    o = sigmoid(splitted_product[2], ctx=ctx)
    g = tanh(splitted_product[3], ctx=ctx)
    ct = add(mul(f, ct_1, ctx=ctx), mul(i, g, ctx=ctx), ctx=ctx)
    ht = mul(o, tanh(ct, ctx=ctx), ctx=ctx)
    return ht, ct
    # END SOLUTION
