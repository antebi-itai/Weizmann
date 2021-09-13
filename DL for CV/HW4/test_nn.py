import unittest  # noqa

import torch  # noqa

from autograd import backward, create_grad_if_necessary
from functional import cross_entropy_loss  # noqa
from nn import LSTMCell, Embedding  # noqa


def check_nonzero_grad(t, name="t"):
  assert isinstance(t.grad, torch.Tensor), f"{name}.grad is not a tensor"
  assert t.grad.size() == t.size(), f"{name}.grad has shape {t.grad.size()}. expected: {t.size()}"
  assert t.grad.abs().max() > 0, f"{name}.grad is all zeros"


class TestEmbeddingLayer(unittest.TestCase):
  def testNumParameters(self):
    embed = Embedding(10, 5)
    num_params = len(embed.parameters())
    expected = 1
    assert num_params == expected, f"Embedding should have {expected} parameters. Got: {num_params}"

  def testContext(self):
    size_x = (3, 4)
    size_w = (10, 5)
    embed = Embedding(size_w[0], size_w[1])
    ctx = []
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    _ = embed(x, ctx=ctx)
    assert ctx, "Embedding should add backward calls to context"

  def testOutput(self):
    size_x = (3, 4)
    size_w = (10, 5)
    embed = Embedding(size_w[0], size_w[1])
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    out = embed(x)

    assert isinstance(out, torch.Tensor), "Embedding outputs should be tensor"
    assert out.size() == (size_x[0], size_x[1], size_w[1]), f"exptected Embedding output to have size ({size_x[0]}, {size_x[1]}, {size_w[1]}). Got: {out.size()}"

  def testBackward(self):
    size_x = (3, 4)
    size_w = (10, 5)
    embed = Embedding(size_w[0], size_w[1])
    ctx = []
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    out = embed(x, ctx=ctx)

    out.grad = torch.randn_like(out)
    backward_fn, args = ctx.pop()
    create_grad_if_necessary(*args)
    backward_fn(*args)

    for i, param in enumerate(embed.parameters()):
      check_nonzero_grad(param, f"param[{i}]")


class TestLSTMCell(unittest.TestCase):
  # def testNumParameters(self):
  #   lstm = LSTMCell(10, 20)
  #   num_params = len(lstm.parameters())
  #   expected = 8
  #   assert num_params == expected, f"LSTMCell should have {expected} parameters. Got: {num_params}"

  def testContext(self):
    lstm = LSTMCell(10, 20)
    ctx = []
    xt = torch.rand(size=(32, 10))
    _ = lstm(xt, ctx=ctx)
    assert ctx, "LSTMCell should add backward calls to context"

  def testOutput(self):
    lstm = LSTMCell(10, 20)
    xt = torch.rand(size=(32, 10))
    out = lstm(xt)
    assert isinstance(out, (tuple, list)) and len(out) == 2, "LSTMCell should return a tuple of two tensors"
    assert all(isinstance(t, torch.Tensor) for t in out), "LSTMCell outputs should be tensors"
    ht, ct = out
    assert ht.size() == (32, 20), f"exptected LSTMCell output `ht` to have size (32, 20). Got: {ht.size()}"
    assert ct.size() == (32, 20), f"exptected LSTMCell output `ct` to have size (32, 20). Got: {ct.size()}"

  def testHtBackwardXt(self):
    lstm = LSTMCell(10, 20)
    ctx = []
    xt = torch.rand(size=(32, 10))
    yt = torch.randint(low=0, high=20, size=(32,))
    ht, ct = lstm(xt, ctx=ctx)
    loss = cross_entropy_loss(ht, yt, ctx=ctx)
    backward(loss, ctx=ctx)
    # for i, param in enumerate(lstm.parameters()):
    #   check_nonzero_grad(param, f"param[{i}]")
    check_nonzero_grad(xt, "xt")

  def testHtBackwardXtHt_1Ct_1(self):
    lstm = LSTMCell(10, 20)
    ctx = []
    xt = torch.rand(size=(32, 10))
    yt = torch.randint(low=0, high=20, size=(32,))
    ht_1 = torch.rand(size=(32, 20))
    ct_1 = torch.rand(size=(32, 20))
    ht, ct = lstm(xt, ht_1, ct_1, ctx=ctx)
    loss = cross_entropy_loss(ht, yt, ctx=ctx)
    backward(loss, ctx=ctx)
    for i, param in enumerate(lstm.parameters()):
      check_nonzero_grad(param, f"param[{i}]")
    check_nonzero_grad(xt, "xt")
    check_nonzero_grad(ht_1, "ht_1")
    check_nonzero_grad(ct_1, "ct_1")

  def testCtBackwardXt(self):
    lstm = LSTMCell(10, 20)
    ctx = []
    xt = torch.rand(size=(32, 10))
    yt = torch.randint(low=0, high=20, size=(32,))
    ht, ct = lstm(xt, ctx=ctx)
    loss = cross_entropy_loss(ht, yt, ctx=ctx)
    backward(loss, ctx=ctx)
    # for i, param in enumerate(lstm.parameters()):
    #   check_nonzero_grad(param, f"param[{i}]")
    check_nonzero_grad(xt, "xt")

  def testCtBackwardXtHt_1Ct_1(self):
    lstm = LSTMCell(10, 20)
    ctx = []
    xt = torch.rand(size=(32, 10))
    yt = torch.randint(low=0, high=20, size=(32,))
    ht_1 = torch.rand(size=(32, 20))
    ct_1 = torch.rand(size=(32, 20))
    ht, ct = lstm(xt, ht_1, ct_1, ctx=ctx)
    loss = cross_entropy_loss(ht, yt, ctx=ctx)
    backward(loss, ctx=ctx)
    for i, param in enumerate(lstm.parameters()):
      check_nonzero_grad(param, f"param[{i}]")
    check_nonzero_grad(xt, "xt")
    check_nonzero_grad(ht_1, "ht_1")
    check_nonzero_grad(ct_1, "ct_1")
