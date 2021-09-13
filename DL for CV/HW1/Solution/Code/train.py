import torch  # noqa

from model import softmax_classifier
from model import softmax_classifier_backward
from model import cross_entropy
from utils import Metric, accuracy  # noqa

__all__ = ['create_model', 'test_epoch', 'test_epoch', 'train_loop']


#################################################
# create_model
#################################################

def create_model():
  """Creates a Softmax Classifier model `(w, b)`.

  Returns:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
  """
  # BEGIN SOLUTION
  num_classes = 10
  in_dim = 28*28
  
  # Initialize W,b with uniform distribution on the interval [0,1)
  w = torch.rand(num_classes, in_dim)
  b = torch.rand(num_classes)
  # Scale & Shift W,b distribution to the interval (-sqrt(k), sqrt(k))
  # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
  sqrt_k = (1 / in_dim)**0.5
  w = (w * (2 * sqrt_k)) - sqrt_k
  b = (b * (2 * sqrt_k)) - sqrt_k
  
  # END SOLUTION
  return w, b


#################################################
# train_epoch
#################################################

def train_epoch(w, b, lr, loader):
  """Trains over an epoch, and returns the accuracy and loss over the epoch.

  Note: The accuracy and loss are average over the epoch. That's different from
  running the classifier over the data again at the end of the epoch, as the
  weights changed over the iterations. However, it's a common practice, since
  iterating over the training set (again) is time and resource exhustive.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    lr (float): The learning rate.
    loader (torch.utils.data.DataLoader): A data loader. An iterator over the dataset.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  device = w.device

  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    # NOTE: In your solution you MUST keep the loss in a tensor called `loss`
    # NOTE: In your solution you MUST keep the acurracy in a tensor called `acc`
    num_classes, in_dim = w.shape
    batch_size = x.shape[0]
    # Reshape the input x
    x = x.reshape(batch_size, in_dim)
    # Run the model to get a prediction
    pred = softmax_classifier(x, w, b)
    # Compute the cross-entropy loss
    loss = cross_entropy(pred, y)
    acc = accuracy(pred, y)
    # Compute the gradients of the weights
    softmax_classifier_backward(x, w, b, pred, y)
    # Update the weights
    w -= lr * w.grad
    b -= lr * b.grad
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# test_epoch
#################################################

def test_epoch(w, b, loader):
  """Evaluating the model at the end of the epoch.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    loader (torch.utils.data.DataLoader): A data loader. An iterator over the dataset.

  Returns:
    acc_metric (Metric): The accuracy metric over the epoch.
    loss_metric (Metric): The loss metric over the epoch.
  """
  device = w.device

  loss_metric = Metric()
  acc_metric = Metric()
  for x, y in loader:
    x, y = x.to(device=device), y.to(device=device)
    # BEGIN SOLUTION
    # NOTE: In your solution you MUST keep the loss in a tensor called `loss`
    # NOTE: In your solution you MUST keep the acurracy in a tensor called `acc`
    num_classes, in_dim = w.shape
    batch_size = x.shape[0]
    # Reshape the input x
    x = x.reshape(batch_size, in_dim)
    # Run the model to get a prediction
    pred = softmax_classifier(x, w, b)
    # Compute the cross-entropy loss
    loss = cross_entropy(pred, y)
    acc = accuracy(pred, y)
    # END SOLUTION
    loss_metric.update(loss.item(), x.size(0))
    acc_metric.update(acc.item(), x.size(0))
  return loss_metric, acc_metric


#################################################
# PROVIDED: train
#################################################

def train_loop(w, b, lr, train_loader, test_loader, epochs, test_every=1):
  """Trains the Softmax Classifier model and report the progress.

  Args:
    w (torch.Tensor): The weight tensor, has shape `(num_classes, in_dim)`.
    b (torch.Tensor): The bias tensor, has shape `(num_classes,)`.
    lr (float): The learning rate.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    test_loader (torch.utils.data.DataLoader): The test set data loader.
    epochs (int): Number of training epochs.
    test_every (int): How frequently to report progress on test data.
  """
  for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_epoch(w, b, lr, train_loader)
    print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
          f'Loss: {train_loss.avg:7.4g}',
          f'Accuracy: {train_acc.avg:.3f}',
          sep='   ')
    if epoch % test_every == 0:
      test_loss, test_acc = test_epoch(w, b, test_loader)
      print(' Test', f'Epoch: {epoch:03d} / {epochs:03d}',
            f'Loss: {test_loss.avg:7.4g}',
            f'Accuracy: {test_acc.avg:.3f}',
            sep='   ')
