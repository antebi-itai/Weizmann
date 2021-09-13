import torch  # noqa

__all__ = ['SGD']


#################################################
# SGD
#################################################

class SGD:
  def __init__(self, parameters, lr):
    """Creates an SGD optimizer.

    Args:
      parameters (List[torch.Tensor]): List of parameters. Each parameter
        should appear at most once.
      lr (float): The learning rate. Should be positive for gradient
        descent.
    """
    if len(set(parameters)) != len(parameters):
      raise ValueError("can't optimize duplicated parameters!")
    # BEGIN SOLUTION
    assert(lr > 0)
    
    self._parameters = parameters
    self._lr = lr
    # END SOLUTION

  def zero_grad(self):
    """Zeros the gradients of all the parameters in the network.

    Note: Gradients are zeroed by setting them to `None`, or by
    zeroing all their values.
    """
    # BEGIN SOLUTION
    for param in self._parameters:
      if param.grad is not None:
        param.grad.zero_()
    # END SOLUTION

  def step(self):
    """Updates the parameter values according to their gradients
    and the learning rate.

    Note: Parameters should be updated in-place.

    Note: The gradients of some parameters might be `None`. You should
    support that case in your solution.
    """
    # BEGIN SOLUTION
    for param in self._parameters:
      param -= self._lr * param.grad
    # END SOLUTION
