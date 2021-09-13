import torch  # noqa
from torch.nn.functional import one_hot  # noqa

__all__ = ['linear', 'relu', 'softmax', 'cross_entropy', 'cross_entropy_loss']


#################################################
# EXAMPLE: mean
#################################################

def mean(x, ctx=None):
  """A differentiable Mean function.

  Backward call:
    backward_fn: mean_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output scalar tensor, the mean of `x`.
  """
  y = x.mean()
  # the backward function with its arguments is appended to `ctx`
  if ctx is not None:
    ctx.append([mean_backward, [y, x]])
  return y


def mean_backward(y, x):
  """Backward computation of `mean`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output scalar tensor.
    x (torch.Tensor): The input tensor.
  """
  # the gradient of `x` is added to `x.grad`
  x.grad += torch.ones_like(x) * (y.grad / x.numel())


#################################################
# linear
#################################################

def linear(x, w, b, ctx=None):
  """A differentiable Linear function. Computes: y = w * x + b

  Backward call:
    backward_fn: linear_backward
    args: y, x, w, b

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
  """
  # VECTORIZATION HINT: torch.mm

  # BEGIN SOLUTION
  y = x.mm(w.t()) + b
  if ctx is not None:
    ctx.append([linear_backward, [y, x, w, b]])
  return y
  # END SOLUTION


def linear_backward(y, x, w, b):
  """Backward computation of `linear`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b`,
  and accumulates them in `x.grad`, `w.grad` and `b.grad` respectively.

  Args:
    y (torch.Tensor): The output tensor, has shape `(batch_size, out_dim)`.
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(out_dim, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(out_dim,)`.
  """
  # VECTORIZATION HINT: torch.mm

  # BEGIN SOLUTION
  x.grad += y.grad.mm(w)
  w.grad += y.grad.t().mm(x)
  b.grad += y.grad.sum(0)
  # END SOLUTION


#################################################
# relu
#################################################

def relu(x, ctx=None):
  """A differentiable ReLU function.

  Note: `y` should be a different tensor than `x`. `x` should not be changed.
        Read about Tensor.clone().

  Note: Don't modify the input in-place.

  Backward call:
    backward_fn: relu_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has non-negative values.
  """
  # BEGIN SOLUTION
  y = x.clamp(min=0)
  if ctx is not None:
    ctx.append([relu_backward, [y, x]])
  return y
  # END SOLUTION


def relu_backward(y, x):
  """Backward computation of `relu`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor. Has non-negative values.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += y.grad * (x > 0).int()
  # END SOLUTION


#################################################
# softmax
#################################################

def softmax(x, ctx=None):
  """A differentiable Softmax function.

  Note: make sure to add `x` from the input to the context,
        and not some intermediate tensor.

  Backward call:
    backward_fn: softmax_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
      Each row in `y` is a probability distribution over the classes.
  """
  # BEGIN SOLUTION
  # Shift input for numerical stability - https://cs231n.github.io/linear-classify/#softmax  
  shifted_x = x - torch.max(x, axis=1, keepdims=True).values
  # Calculate Softmax
  exp_shifted_x = torch.exp(shifted_x)
  y = exp_shifted_x / torch.sum(exp_shifted_x, dim=1, keepdims=True)

  if ctx is not None:
    ctx.append([softmax_backward, [y, x]])
  return y
  # END SOLUTION


def softmax_backward(y, x):
  """Backward computation of `softmax`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor, has shape `(batch_size, num_classes)`.
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.einsum

  # BEGIN SOLUTION
  batch_size, num_classes = y.shape
  ## calculate d(yi)/d(xj) - over the entire batch
  # correct only for i != j:
  dy_dx = - y.unsqueeze(2) @ y.unsqueeze(1)
  # add correction for i == j:
  dy_dx += torch.diag_embed(y)

  # chain rule
  x.grad += (y.grad.unsqueeze(1) @ dy_dx).squeeze()
  # END SOLUTION


#################################################
# cross_entropy
#################################################

def cross_entropy(pred, target, ctx=None):
  """A differentiable Cross-Entropy function for hard-labels.

  Backward call:
    backward_fn: cross_entropy
    args: loss, pred, target

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
  """
  # VECTORIZATION HINT: one_hot, torch.gather
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION
  batch_size, num_classes = pred.shape

  safe_pred = pred.clamp(min=eps) # for numerical stability
  correct_class_pred = (one_hot(target, num_classes) * safe_pred).sum(1)
  loss = - correct_class_pred.log()

  if ctx is not None:
    ctx.append([cross_entropy_backward, [loss, pred, target]])

  return loss
  # END SOLUTION


def cross_entropy_backward(loss, pred, target):
  """Backward computation of `cross_entropy`.

  Propagates the gradients of `loss` (in `loss.grad`) to `pred`,
  and accumulates them in `pred.grad`.

  Note: `target` is an integer tensor and has no gradients.

  Args:
    loss (torch.Tensor): The per-example cross-entropy tensor, has shape `(batch_size,).
      Each value is the cross-entropy loss of that example in the batch.
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Each row is a probability distribution over the classes.
    target (torch.Tensor): The tragets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
  """
  # VECTORIZATION HINT: one_hot, torch.gather, torch.scatter_add
  eps = torch.finfo(pred.dtype).tiny
  # BEGIN SOLUTION
  batch_size, num_classes = pred.shape
  
  # calculate d(loss)/d(pred) - over the entire batch
  safe_pred = pred.clamp(min=eps) # for numerical stability
  dloss_dpred = (- 1 / safe_pred) * one_hot(target, num_classes)
  # chain rule
  pred.grad += dloss_dpred * loss.grad.unsqueeze(1)
  # END SOLUTION


#################################################
# PROVIDED: cross_entropy_loss
#################################################

def cross_entropy_loss(pred, target, ctx=None):
  """A differentiable Cross-Entropy loss for hard-labels.

  This differentiable function is similar to PyTorch's cross-entropy function.

  Note: Unlikehis function expects `pred` to be BEFORE softmax.

  Note: You should not implement the backward of that function explicitly, as you use only
        differentiable functions for that. That part of the "magic" in autograd --
        you can simply compose differentiable functions, and it works!

  Args:
    pred (torch.Tensor): The predictions tensor, has shape `(batch_size, num_classes)`.
      Unlike `cross_entropy`, this prediction IS NOT a probability distribution over
      the classes. It expects to see predictions BEFORE sofmax.
    target (torch.Tensor): The targets integer tensor, has shape `(batch_size,)`.
      Each value is the index of the correct class.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    loss (torch.Tensor): The scalar loss tensor. The mean loss over the batch.
  """
  pred = softmax(pred, ctx=ctx)
  batched_loss = cross_entropy(pred, target, ctx=ctx)
  loss = mean(batched_loss, ctx=ctx)
  return loss
