import torch
import utils
import models
import os

from autograd import backward  # noqa
from utils import Metric  # noqa
from optim import MomentumSGD # noqa
from functional  import cross_entropy_loss



def train_epoch(model, criterion, optimizer, loader, device):
  """Trains over an epoch, and returns the  loss over the epoch.

  Note: The loss is average over the epoch. That's different from
  running the classifier over the data again at the end of the epoch, as the
  weights changed over the iterations. However, it's a common practice, since
  iterating over the training set (again) is time and resource exhustive.

  Note: You MUST have `loss` tensor with the loss value.

  Args:
    model (nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (optim.Optimizer): The optimizer.
    loader (torch.utils.data.DataLoader): The training set data loader.
    device (torch.device): The device to run on.

  Returns:
    loss_metric (Metric): The loss metric over the epoch.
  """
  batch_size = loader.batch_size
  loss_metric = Metric()
  for x, y in loader:
    # make sure you iterate over completely full batches, only
    if x.shape[0] != batch_size:
        break

    x, y = x.to(device=device), y.to(device=device)
    ctx = []
    optimizer.zero_grad()
    pred = model(x, ctx=ctx)
    loss = criterion(pred, y, ctx=ctx)
    backward(loss, ctx)
    optimizer.step() # TODO add gradient clipping
    loss_metric.update(loss.item(), x.size(0))

  return loss_metric


def train_loop(model, criterion, optimizer, train_loader, device, epochs):
  """Trains a model to minimize some loss function and reports the progress.

  Args:
    model (nn.Module): The model.
    criterion (callable): The loss function. Should return a scalar tensor.
    optimizer (optim.MomentumSGD): The optimizer.
    train_loader (torch.utils.data.DataLoader): The training set data loader.
    device (torch.device): The device to run on.
    epochs (int): Number of training epochs.
  """
  for epoch in range(1, epochs + 1):
    train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
    print('Train', f'Epoch: {epoch:03d} / {epochs:03d}',
          f'Loss: {train_loss.avg:7.4g}',
          sep='   ')