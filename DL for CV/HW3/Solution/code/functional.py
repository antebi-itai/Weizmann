import math  # noqa

import torch  # noqa
from torch.nn.functional import one_hot  # noqa
from torch.nn.modules.utils import _pair, _ntuple
from torch.nn.functional import unfold, fold

from hw2_functional import *  # noqa
from hw2_functional import __all__ as __old_all__

__new_all__ = ['view', 'add', 'conv2d', 'max_pool2d']
__all__ = __old_all__ + __new_all__ # 


#################################################
# conv2d
#################################################

def conv2d(x, w, b=None, padding=0, stride=1, dilation=1, groups=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d

  Backward call:
    backward_fn: conv2d_backward
    args: y, x, w, b, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_height, out_width)`.
  """
  assert w.size(0) % groups == 0, \
    f'expected w.size(0)={w.size(0)} to be divisible by groups={groups}'
  assert x.size(1) % groups == 0, \
    f'expected x.size(1)={x.size(1)} to be divisible by groups={groups}'
  assert x.size(1) // groups == w.size(1), \
    f'expected w.size(1)={w.size(1)} to be x.size(1)//groups={x.size(1)}//{groups}'

  # BEGIN SOLUTION
  # extract and parse input
  if b is None:
    b = torch.zeros(out_channels, dtype=x.dtype, device=x.device)
  batch_size, _, height, width = x.shape
  out_channels, _, kernel_height, kernel_width = w.shape
  in_channels = w.shape[1] * groups
  padding_h, padding_w = _pair(padding) if type(padding) is int else padding
  stride_h, stride_w = _pair(stride) if type(stride) is int else stride
  dilation_h, dilation_w = _pair(dilation) if type(dilation) is int else dilation
  out_height = int(1 + ((height + 2*padding_h - dilation_h * (kernel_height - 1) - 1) / (stride_h)))
  out_width = int(1 + ((width + 2*padding_w - dilation_w * (kernel_width - 1) - 1) / (stride_w)))
  
  # unfold x and split to groups
  patches = unfold(x, (kernel_height, kernel_width),
                    dilation=dilation,
                    padding=padding, 
                    stride=stride)
  patches = patches.reshape(batch_size, in_channels, kernel_height*kernel_width, out_height*out_width)
  patches_group_split = torch.stack(patches.chunk(chunks=groups, dim=1), dim=1)
  patches_group_split = patches_group_split.reshape(batch_size, groups, int(in_channels/groups)*kernel_height*kernel_width, out_height*out_width)
  
  # split w to groups
  w_group_split = w.reshape(out_channels, int(in_channels/groups)*kernel_height*kernel_width)
  w_group_split = torch.stack(w_group_split.chunk(chunks=groups, dim=0))
  w_group_split = w_group_split.reshape(1, groups, int(out_channels/groups), int(in_channels/groups)*kernel_height*kernel_width)
  
  # compute convolution as matrix multiplication
  convolved_patches = w_group_split @ patches_group_split
  convolved_patches = convolved_patches.reshape(batch_size, out_channels, out_height, out_width)
  convolved_patches += b.reshape(1, -1, 1, 1)

  y = convolved_patches

  if ctx is not None:
    ctx += [(conv2d_backward, [y, x, w, b, padding, stride, dilation, groups])]
  
  return y
  # END SOLUTION


def conv2d_backward(y, x, w, b, padding, stride, dilation, groups):
  """Backward computation of `conv2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, `w` and `b` (if `b` is not None),
  and accumulates them in `x.grad`, `w.grad` and `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, out_channels, out_height, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    w (torch.Tensor): The weight tensor.
      Has shape `(out_channels, in_channels, kernel_height, kernel_width)`.
    b (torch.Tensor): The bias tensor. Has shape `(out_channels,)`.
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    groups (int, Optional): Number of groups. Defaults to 1.
  """
  # BEGIN SOLUTION
  # extract and parse input
  batch_size, _, height, width = x.shape
  out_channels, _, kernel_height, kernel_width = w.shape
  in_channels = w.shape[1] * groups
  padding_h, padding_w = _pair(padding) if type(padding) is int else padding
  stride_h, stride_w = _pair(stride) if type(stride) is int else stride
  dilation_h, dilation_w = _pair(dilation) if type(dilation) is int else dilation
  out_height = int(1 + ((height + 2*padding_h - dilation_h * (kernel_height - 1) - 1) / (stride_h)))
  out_width = int(1 + ((width + 2*padding_w - dilation_w * (kernel_width - 1) - 1) / (stride_w)))
  
  # unfold x and split to groups
  patches = unfold(x, (kernel_height, kernel_width),
                    dilation=dilation,
                    padding=padding, 
                    stride=stride)
  patches = patches.reshape(batch_size, in_channels, kernel_height*kernel_width, out_height*out_width)
  patches = patches.permute([0, 3, 1, 2])
  patches_group_split = torch.stack(patches.chunk(chunks=groups, dim=2), dim=0)
  patches_group_split = patches_group_split.reshape(groups, batch_size*out_height*out_width, int(in_channels/groups)*kernel_height*kernel_width)

  # b.grad
  b.grad += y.grad.sum([0, 2, 3])

  # w.grad
  y_grad_permuted = y.grad.permute([1, 0, 2, 3])
  y_grad_permuted = y_grad_permuted.reshape(out_channels, batch_size*out_height*out_width)
  y_grad_permuted_group_split = torch.stack(y_grad_permuted.chunk(chunks=groups, dim=0), dim=0)
  w_grad_permuted = y_grad_permuted_group_split.matmul(patches_group_split)
  w.grad += w_grad_permuted.reshape(out_channels, int(in_channels/groups), kernel_height, kernel_width)

  # patches.grad (unfolded x)
  y_grad_permuted_group_split = y_grad_permuted_group_split.permute([0, 2, 1])
  w_permuted = w.reshape(out_channels, -1)
  w_permuted_group_split = torch.stack(w_permuted.chunk(chunks=groups, dim=0), dim=0)
  patches_grad_group_split = y_grad_permuted_group_split.matmul(w_permuted_group_split)

  # x.grad (fold the reshaped patches.grad)
  patches_grad_group_split = patches_grad_group_split.reshape(groups, batch_size*out_height*out_width, int(in_channels/groups), kernel_height*kernel_width)
  patches_grad_group_split = patches_grad_group_split.permute([1, 0, 2, 3])
  patches_grad_group_split = patches_grad_group_split.reshape(batch_size, out_height*out_width, in_channels*kernel_height*kernel_width)
  patches_grad_group_split = patches_grad_group_split.permute([0, 2, 1])
  x.grad += fold(patches_grad_group_split, 
                output_size=(height, width), 
                kernel_size=(kernel_height, kernel_width),
                dilation=dilation,
                padding=padding, 
                stride=stride)
  # END SOLUTION


#################################################
# max_pool2d
#################################################

def max_pool2d(x, kernel_size, padding=0, stride=1, dilation=1, ctx=None):
  """A differentiable convolution of 2d tensors.

  Note: Read this following documentation regarding the output's shape.
  https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html#torch.nn.MaxPool2d

  Backward call:
    backward_fn: max_pool2d_backward
    args: y, x, padding, stride, dilation

  Args:
    x (torch.Tensor): The input tensor. Has shape `(batch_size, in_channels, height, width)`.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_height, out_width)`.
  """
  # BEGIN SOLUTION
  # extract and parse input
  batch_size, in_channels, height, width = x.shape
  kernel_height, kernel_width = _pair(kernel_size) if type(kernel_size) is int else kernel_size
  padding_h, padding_w = _pair(padding) if type(padding) is int else padding
  stride_h, stride_w = _pair(stride) if type(stride) is int else stride
  dilation_h, dilation_w = _pair(dilation) if type(dilation) is int else dilation
  out_height = int(1 + ((height + 2*padding_h - dilation_h * (kernel_height - 1) - 1) / (stride_h)))
  out_width = int(1 + ((width + 2*padding_w - dilation_w * (kernel_width - 1) - 1) / (stride_w)))

  # unfold x to patches
  # patches.shape - (batch_size, C*Kh*Kw, out_height*out_width)
  patches = unfold(x, (kernel_height, kernel_width),
                    dilation=dilation,
                    padding=padding, 
                    stride=stride)
  patches = patches.reshape(batch_size, in_channels, kernel_height*kernel_width, out_height, out_width)

  # take max over each patch
  y, index = patches.max(dim=2)

  if ctx is not None:
    ctx += [(max_pool2d_backward, [y, x, index, kernel_size, padding, stride, dilation])]

  return y
  # END SOLUTION


def max_pool2d_backward(y, x, index, kernel_size, padding, stride, dilation):
  """Backward computation of `max_pool2d`.

  Propagates the gradients of `y` (in `y.grad`) to `x` and accumulates it in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
      Has shape `(batch_size, in_channels, out_height, out_width)`.
    x (torch.Tensor): The input tensor.
      Has shape `(batch_size, in_channels, height, width)`.
    index (torch.Tensor): Auxilary tensor with indices of the maximum elements. You are
      not restricted to a specific format.
    kernel_size (Tuple[int, int] or int): The kernel size in each dimension (height, width).
    padding (Tuple[int, int] or int, Optional): The padding in each dimension (height, width).
      Defaults to 0.
    stride (Tuple[int, int] or int, Optional): The stride in each dimension (height, width).
      Defaults to 1.
    dilation (Tuple[int, int] or int, Optional): The dilation in each dimension (height, width).
      Defaults to 1.
  """
  # BEGIN SOLUTION
  # extract and parse input
  batch_size, in_channels, height, width = x.shape
  batch_size, in_channels, out_height, out_width = y.shape
  kernel_height, kernel_width = _pair(kernel_size) if type(kernel_size) is int else kernel_size

  one_hot_entries = one_hot(index, num_classes=kernel_height*kernel_width)
  one_hot_entries = one_hot_entries.permute([0, 1, 4, 2, 3])
  permuted_x_grad = one_hot_entries * y.grad.unsqueeze(2)
  permuted_x_grad = permuted_x_grad.reshape(batch_size, in_channels*kernel_height*kernel_width, out_height*out_width)
  x.grad += fold(permuted_x_grad, 
                output_size=(height, width), 
                kernel_size=(kernel_height, kernel_width),
                dilation=dilation,
                padding=padding, 
                stride=stride)
  # END SOLUTION


#################################################
# view
#################################################

def view(x, size, ctx=None):
  """A differentiable view function.

  Backward call:
    backward_fn: view_backward
    args: y, x

  Args:
    x (torch.Tensor): The input tensor.
    size (torch.Size): The new size (shape).
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. Has shape `size`.
  """
  # BEGIN SOLUTION
  y = x.view(size)
  
  if ctx is not None:
    ctx += [(view_backward, [y, x])]
  
  return y
  # END SOLUTION


def view_backward(y, x):
  """Backward computation of `view`.

  Propagates the gradients of `y` (in `y.grad`) to `x`, and accumulates them in `x.grad`.

  Args:
    y (torch.Tensor): The output tensor.
    x (torch.Tensor): The input tensor.
  """
  # BEGIN SOLUTION
  x.grad += y.grad.view(x.shape)
  # END SOLUTION


#################################################
# add
#################################################

def add(a, b, ctx=None):
  """A differentiable addition of two tensors.

  Backward call:
    backward_fn: add_backward
    args: y, a, b

  Args:
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor. Should have the same shape as `a`.
    ctx (List, optional): The autograd context. Defaults to None.

  Returns:
    y (torch.Tensor): The output tensor. The sum of `a + b`.
  """
  assert a.size() == b.size(), 'tensors should have the same size'
  # BEGIN SOLUTION
  y = a + b
  
  if ctx is not None:
    ctx += [(add_backward, [y, a, b])]
  
  return y
  # END SOLUTION
  

def add_backward(y, a, b):
  """Backward computation of `add`.

  Propagates the gradients of `y` (in `y.grad`) to `a` and `b`, and accumulates them in `a.grad`,
  `b.grad`, respectively.

  Args:
    y (torch.Tensor): The output tensor.
    a (torch.Tensor): The first input tensor.
    b (torch.Tensor): The second input tensor.
  """
  # BEGIN SOLUTION
  a.grad += y.grad
  b.grad += y.grad
  # END SOLUTION
