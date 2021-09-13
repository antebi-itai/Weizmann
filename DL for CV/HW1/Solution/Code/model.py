import torch
from torch.nn.functional import one_hot

__all__ = [
  'softmax',
  'cross_entropy',
  'softmax_classifier',
  'softmax_classifier_backward'
]


##########################################################
# Softmax
##########################################################

def softmax(x):
  """Softmax activation.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, num_classes)`.

  Returns:
    y (torch.Tensor): The softmax distribution over `x`. Has the same shape as `x`.
      Each row in `y` is a probability over the classes.
  """
  # BEGIN SOLUTION
  # Shift input for numerical stability - https://cs231n.github.io/linear-classify/#softmax  
  x -= torch.max(x, axis=1, keepdims=True).values
  # Calculate Softmax
  exp_x = torch.exp(x)
  y = exp_x / torch.sum(exp_x, dim=1, keepdims=True)
  return y
  # END SOLUTION


##########################################################
# Cross Entropy
##########################################################

def cross_entropy(pred, target):
  """Cross-entropy loss for hard-labels.

  Hint: You can use the imported `one_hot` function.

  Args:
    pred (torch.Tensor): The predictions (probability per class), has shape `(batch_size, num_classes)`.
    target (torch.Tensor): The target classes (integers), has shape `(batch_size,)`.

  Returns:
    loss (torch.Tensor): The mean cross-entropy loss over the batch.
  """
  # BEGIN SOLUTION
  batch_size, num_classes = pred.shape
  # Find the predicted probability of the correct class of each instance in the batch
  correct_class_locations = one_hot(target, num_classes=num_classes)
  correct_class_probabilities = torch.sum(pred * correct_class_locations, dim=1)
  # Clamp for numerical stability - avoid infinite loss
  eps = 1e-7
  stable_correct_class_probabilities = correct_class_probabilities.clamp(eps, 1)
  # Calculate loss of every instance in the batch
  instance_losses = -torch.log(stable_correct_class_probabilities)
  # The loss of the entire batch is the average of losses of all instances in batch
  loss = torch.mean(instance_losses)
  return loss
  # END SOLUTION


##########################################################
# Softmax Classifier
##########################################################

def softmax_classifier(x, w, b):
  """Applies the prediction of the Softmax Classifier.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.

  Returns:
    pred (torch.Tensor): The predictions, has shape `(batch_size, num_classes)`.
      Each row is a probablity measure over the classes.
  """
  # BEGIN SOLUTION
  pred = softmax(x.mm(w.t()) + b)
  return pred
  # END SOLUTION


##########################################################
# Softmax Classifier Backward
##########################################################

def softmax_classifier_backward(x, w, b, pred, target):
  """Computes the gradients of weight in the Softmax Classifier.

  The gradients computed for the parameters `w` and `b` should be stored in
  `w.grad` and `b.grad`, respectively.

  Hint: You can use the imported `one_hot` function.

  Args:
    x (torch.Tensor): The input tensor, has shape `(batch_size, in_dim)`.
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    pred (torch.Tensor): The predictions (probability per class), has shape `(batch_size, num_classes)`.
    target (torch.Tensor): The target classes (integers), has shape `(batch_size,)`.
  """
  # BEGIN SOLUTION
  batch_size, num_classes = pred.shape

  # Calculate gradient of W
  pred_target_diff = pred - one_hot(target, num_classes=num_classes)
  w.grad = pred_target_diff.t().mm(x) / batch_size
  
  # Calculate gradient of b
  b.grad = torch.mean(pred_target_diff, dim=0)
  # END SOLUTION
