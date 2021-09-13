import torch  # noqa
import torchvision  # noqa

__all__ = ['Metric', 'accuracy', 'load_mnist']


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
    acc (torch.Tensor): A scalar tensor with mean accuracy, i.e. % of correct predictions.
  """
  acc = (pred.argmax(dim=1) == target)
  return acc.to(pred).mean()


#################################################
# PROVIDED: load_mnist
#################################################

def load_mnist(root='/content/data', mode='train'):
  # bugfix: https://github.com/pytorch/vision/issues/1938
  mode = mode.lower()
  assert mode in ('train', 'test', 'val', 'eval')
  import urllib
  opener = urllib.request.build_opener()
  opener.addheaders = [('User-agent', 'Mozilla/5.0')]
  urllib.request.install_opener(opener)
  torchvision.datasets.MNIST.resources = [
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
            ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
  ]

  # load dataset
  return torchvision.datasets.MNIST(
      root=root,
      train=mode == 'train',
      download=True,
      transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
      ])
  )
