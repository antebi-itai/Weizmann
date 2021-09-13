import math  # noqa
from typing import Sequence  # noqa

import torch  # noqa

from autograd import create_grad_if_necessary
from hw3_functional import linear, add
from hw3_functional import *  # noqa
from hw3_functional import __all__ as __old_all__

__all__ = ['sigmoid', 'tanh', 'cat', 'mul', 'unbind', 'embedding']
__all__ += __old_all__


###########################################################
# activations: sigmoid, tanh
###########################################################

def sigmoid(x, ctx=None):
  """A differentiable sigmoid activation.

  Formula:
    y = 1 / (1 + exp(-x))

  Notes:
    1. `y` should be a different tensor than `x`.
    2. Don't modify the input in-place.
    3. Your implementation should be numerically stable.
    4. You should not use `torch.sigmoid` or similar function.

  Backward call:
    backward_fn: sigmoid_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
  """
  # BEGIN SOLUTION
  y = 1 / (1 + torch.exp(-x))
  if ctx is not None:
    ctx.append([sigmoid_backward, [y, x]])
  return y
  # END SOLUTION


def sigmoid_backward(y, x):
  """Backward computation of `sigmoid`.

  Propagates the gradient of `y` (in `y.grad`) to `x`, and accumulates it
  in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += y.grad * (torch.exp(-x) * y.pow(2))
  # END SOLUTION


def tanh(x, ctx=None):
  """A differentiable hyperbolic tangent activation.

  Formula:
    y = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

  Notes:
    1. `y` should be a different tensor than `x`.
    2. Don't modify the input in-place.
    3. Your implementation should be numerically stable.
    4. You should not use `torch.tanh` or similar function.

  Backward call:
    backward_fn: tanh_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
  """
  # BEGIN SOLUTION
  y = torch.empty_like(x)
  y_pos = ((1 - torch.exp(-2*x)) / (1 + torch.exp(-2*x)))
  y_neg = ((torch.exp(2*x) - 1) / (torch.exp(2*x) + 1))
  y[x >= 0] = y_pos[x >= 0]
  y[x < 0] = y_neg[x < 0]
  if ctx is not None:
    ctx.append([tanh_backward, [y, x]])
  return y
  # END SOLUTION


def tanh_backward(y, x):
  """Backward computation of `tanh`.

  Propagates the gradient of `y` (in `y.grad`) to `x`, and accumulates it
  in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += y.grad * (1 - y.pow(2))
  # END SOLUTION


###########################################################
# utilities: cat, unbind
###########################################################

def cat(tensors, dim=0, ctx=None):
  """A differentiable concatenation function.

  Concatentates multiple tensors along a dimension `dim`.

  Notes:
    1. Your may use `torch.cat`.

  Backward call:
    backward_fn: cat_backward
    args: y, tensors, dim

  Args:
    tensors (Sequence[torch.Tensor]): A sequence of tensors to concatenate.
    dim (int, optional): The dimension to concatenate along. Defaults to 0.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The concatentated output tensor.
  """
  assert isinstance(tensors, Sequence)
  # BEGIN SOLUTION
  y = torch.cat(tensors, dim=dim)
  if ctx is not None:
    ctx.append([cat_backward, [y, tensors, dim]])
  return y
  # END SOLUTION


def cat_backward(y, tensors, dim):
  """Backward computation of `cat`.

  Propagates the gradient of `y` (in `y.grad`) to the tensors in `tensors`,
  and accumulates them in `t.grad` for each `t` in `tensors`.

  Args:
    y (torch.Tensor): The concatentated output tensor.
    tensors (Sequence[torch.Tensor]): A sequence of tensors to concatenate.
    dim (int): The dimension to concatenate along.
  """
  create_grad_if_necessary(*tensors)
  # BEGIN SOLUTION
  tensors_sizes = [tensor.shape[dim] for tensor in tensors]
  y_grads = y.grad.split(tensors_sizes, dim=dim)
  for tensor, y_grad in zip(tensors, y_grads):
    tensor.grad += y_grad
  # END SOLUTION


def unbind(x, dim=0, ctx=None):
  """A differentiable unbind function.

  Unbind a tensor `x` into a tuple of tensors along a dimension `dim`.

  Notes:
    1. You may use `torch.unbind`.

  Backward call:
    backward_fn: unbind_backward
    args: y_tensors, x, dim

  Args:
    x (torch.Tensor): A tensor to unbind.
    dim (int, optional): The dimension to unbind along. Defaults to 0.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y_tensors (Sequence[torch.Tensor]): A tuple of tensors.
  """
  # BEGIN SOLUTION
  y_tensors = x.unbind(dim=dim)
  if ctx is not None:
    ctx.append([unbind_backward, [y_tensors, x, dim]])
  return y_tensors
  # END SOLUTION


def unbind_backward(y_tensors, x, dim):
  """Backward computation of `unbind`.

  Propagates the gradients of `y_tensors` (in `y.grad` for each `y` in `y_tensors`),
  to `x` and accumulates it in `x.grad`.

  Notes:
    1. You may use `torch.stack`.

  Args:
    y_tensors (Sequence[torch.Tensor]): A tuple of tensors.
    x (torch.Tensor): A tensor to unbind.
    dim (int): The dimension to unbind along.
  """
  create_grad_if_necessary(*y_tensors)
  # BEGIN SOLUTION
  x.grad += torch.stack([y_tensor.grad for y_tensor in y_tensors], dim=dim)
  # END SOLUTION


###########################################################
# utilities: mul, embedding
###########################################################

def mul(a, b, ctx=None):
  """A differentiable hadamard multiplication.

  Formula:
    y = a * b

  Notes:
    1. Don't modify the input in-place.

  Backward call:
    backward_fn: mul_backward
    args: y, a, b

  Args:
    a (torch.Tensor): The first tensor argument.
    b (torch.Tensor): The second tensor argument.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, hadamard product of the inputs.
  """
  # BEGIN SOLUTION
  y = a * b
  if ctx is not None:
    ctx.append([mul_backward, [y, a, b]])
  return y
  # END SOLUTION


def mul_backward(y, a, b):
  """Backward computation of `mul`.

  Propagates the gradient of `y` (in `y.grad`) to `a` and `b`, and accumulates
  them in `a.grad` and `b.grad`, respsectively.

  Args:
    y (torch.Tensor): The output tensor, hadamard product of the inputs.
    a (torch.Tensor): The first tensor argument.
    b (torch.Tensor): The second tensor argument.
  """
  # BEGIN SOLUTION
  a.grad += y.grad * b
  b.grad += y.grad * a
  # END SOLUTION


def embedding(x, w, ctx=None):
  """A differentiable embedding.
  Map each entry of x to an embedding stored in W.

  Formula:
    for each entry of x return W[x[i][j], :]

  Backward call:
    backward_fn: mul_backward
    args: y, x, w

  Args:
    x (torch.LongTensor): The input tensor. Has shape `(batch_size, sequence_length)`.
      x entries must be b non-negative and smaller than the vocabulary size.
    w (torch.Tensor): The embedding matrix. Has shape `(vocab_size, embedding_size)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The embedding tensor, Has shape `(batch_size, sequence_length, embedding_size)`.
  """
  # BEGIN SOLUTION
  y = w[x]
  if ctx is not None:
    ctx.append([embedding_backward, [y, x, w]])
  return y
  # END SOLUTION


def embedding_backward(y, x, w):
  """Backward computation of `embedding`.

  Propagates the gradients of `y` (in `y.grad`) to `w`, and accumulates
  them in `w.grad.

  Args:
    y (torch.Tensor): The output tensor, embedding of the inputs.
    x (torch.Tensor): The input tensor argument.
    w (torch.Tensor): The embedding matrix argument.
  """
  # HINT: See pytorch index_add_
  # BEGIN SOLUTION
  batch_size, sequence_length = x.shape
  _, _, embedding_size = y.shape
  x_vectorized = x.reshape(batch_size * sequence_length)
  y_grad_vectorized = y.grad.reshape(batch_size * sequence_length, embedding_size)
  w.grad.index_add_(dim=0, index=x_vectorized, source=y_grad_vectorized)
  # END SOLUTION

