import torch  # noqa

from nn import Module, LSTMCell, Embedding, Linear  # noqa
from functional import relu, view, unbind  # noqa

__all__ = ['LSTM']


class LSTM(Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim):
    """ Initialize LSTM network."""
    super().__init__()
    # BEGIN SOLUTION
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim

    self.embed = Embedding(vocab_size=self.vocab_size, embedding_dim=self.embedding_dim)
    self.lstm = LSTMCell(in_dim=self.embedding_dim, hidden_dim=self.hidden_dim)
    self.linear = Linear(self.hidden_dim, self.vocab_size)
    self._modules += [ 'embed', 'lstm', 'linear']
    # END SOLUTION

  def forward(self, x, ctx=None):
    """Computes the forward function of the network.

    Args:
      x (torch.LongTensor): The input tensor. Has shape `(batch_size, sequence_length)`.
      ctx (List, optional): The autograd context. Defaults to None.

    Returns:
      p (torch.Tensor): The probability tensor. Has shape `(batch_size, vocabulary_size)`.
    """
    assert x.dim() == 2, f"x should be a batch of sequences (2D). got: {x.dim()}"
    # BEGIN SOLUTION
    batch_size, sequence_length = x.shape
    embedded_x = self.embed(x, ctx=ctx)
    sequence = unbind(embedded_x, dim=1, ctx=ctx)
    ht, ct = None, None
    for word in sequence:
      ht, ct = self.lstm(word, ht_1=ht, ct_1=ct, ctx=ctx)
    p = self.linear(ht, ctx=ctx)
    return p
    # END SOLUTION
