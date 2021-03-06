import torch  # noqa
import torchvision  # noqa

__all__ = ['Metric', 'accuracy', 'load_mnist', 'load_cifar10']


#################################################
# PROVIDED: Metric
#################################################

class Metric:
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt


#################################################
# PROVIDED: accuracy
#################################################

def accuracy(pred, target):
  """Computes accuracy of a multiclass classification task.

  Args:
    pred (torch.Tensor): Tensor of predictions. Has shape `(batch_size, num_classes)`.
    target ([type]): Integer tensor of target classes (correct labels). Has shape `(batch_size,)`.


  Returns:
    acc (torch.Tensor): A scalar tensor with mean accuracy, i.e. percentage of correct predictions.
  """
  acc = (pred.argmax(dim=1) == target)
  return acc.to(pred).mean()


#################################################
# PROVIDED: load_mnist
#################################################

def load_mnist(root='./data', mode='train'):
  mode = mode.lower()
  assert mode in ('train', 'test', 'val', 'eval')

  # load dataset
  transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
  return torchvision.datasets.MNIST(
    root=root,
    train=mode == 'train',
    download=True,
    transform=transform
  )

def load_cifar10(root='./data', mode='train'):
  mode = mode.lower()
  assert mode in ('train', 'test', 'val', 'eval')

  # load dataset
  transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
  return torchvision.datasets.CIFAR10(
    root=root,
    train=mode == 'train',
    download=True,
    transform=transform
  )
