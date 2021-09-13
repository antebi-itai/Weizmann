import unittest

import torch

# import functional
from functional import sigmoid, tanh, mul, cat, unbind, embedding  # noqa


def assert_tensor_in(tensor, lst, msg=''):
  assert any(tensor is x for x in lst), msg


class TestSigmoid(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    size = (3, 4)
    x = torch.rand(size=size, dtype=self.dtype)
    ctx = [None]
    x = torch.rand(size=size, dtype=self.dtype)
    y = sigmoid(x, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")

  def _test_forward(self, size, bounds):
    x = -bounds + (2 * bounds) * torch.rand(size=size, dtype=self.dtype)
    y = sigmoid(x)
    y_ = torch.sigmoid(x)
    dbg = f'x: {x}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size, bounds):
    x = -bounds + (2 * bounds) * torch.rand(size=size, dtype=self.dtype)
    ctx = []
    x.grad = torch.zeros_like(x)
    y = sigmoid(x, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      y_ = torch.sigmoid(x_)
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'x: {x}\ny.grad: {y.grad}'
    dbg_x = dbg + f'\ngot: {x.grad}\nexpected: {x_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_x)

  def testForward(self):
    self._test_forward(size=(), bounds=10)

  def testForwardND(self):
    self._test_forward(size=(2, 3, 4), bounds=10)

  def testBackward(self):
    self._test_backward(size=(), bounds=10)

  def testBackwardND(self):
    self._test_backward(size=(2, 3, 4), bounds=10)

  def testForwardStability(self):
    b = 10000
    x = torch.tensor([b, -b], dtype=self.dtype)
    y = sigmoid(x)
    y_ = torch.sigmoid(x)
    dbg = f'x: {x}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)


class TestTanh(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    size = (3, 4)
    x = torch.rand(size=size, dtype=self.dtype)
    ctx = [None]
    x = torch.rand(size=size, dtype=self.dtype)
    y = tanh(x, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")

  def _test_forward(self, size, bounds):
    x = -bounds + (2 * bounds) * torch.rand(size=size, dtype=self.dtype)
    y = tanh(x)
    y_ = torch.tanh(x)
    dbg = f'x: {x}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size, bounds):
    x = -bounds + (2 * bounds) * torch.rand(size=size, dtype=self.dtype)
    ctx = []
    x.grad = torch.zeros_like(x)
    y = tanh(x, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      y_ = torch.tanh(x_)
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'x: {x}\ny.grad: {y.grad}'
    dbg_x = dbg + f'\ngot: {x.grad}\nexpected: {x_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_x)

  def testForward(self):
    self._test_forward(size=(), bounds=10)

  def testForwardND(self):
    self._test_forward(size=(2, 3, 4), bounds=10)

  def testBackward(self):
    self._test_backward(size=(), bounds=10)

  def testBackwardND(self):
    self._test_backward(size=(2, 3, 4), bounds=10)

  def testForwardStability(self):
    b = 10000
    x = torch.tensor([b, -b], dtype=self.dtype)
    y = tanh(x)
    y_ = torch.tanh(x)
    dbg = f'x: {x}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)


class TestMul(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    size = (3, 4)
    ctx = [None]
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    y = mul(a, b, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(a, backward_args, "a isn't in backward_args")
    assert_tensor_in(b, backward_args, "b isn't in backward_args")

  def _test_forward(self, size):
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    y = mul(a, b)
    y_ = a * b
    dbg = f'a: {a}\nb: {b}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size):
    a = torch.rand(size=size, dtype=self.dtype)
    b = torch.rand(size=size, dtype=self.dtype)
    ctx = []
    a.grad = torch.zeros_like(a)
    b.grad = torch.zeros_like(b)
    y = mul(a, b, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      a_ = a.detach().clone().requires_grad_()
      b_ = b.detach().clone().requires_grad_()
      y_ = a_ * b_
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'a: {a}\nb: {b}\ny.grad: {y.grad}'
    dbg_a = dbg + f'\ngot (a.grad): {a.grad}\nexpected (a.grad): {a_.grad}'
    dbg_b = dbg + f'\ngot (b.grad): {b.grad}\nexpected (b.grad): {b_.grad}'
    torch.testing.assert_allclose(a.grad, a_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_a)
    torch.testing.assert_allclose(b.grad, b_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_b)

  def testForward(self):
    self._test_forward(size=())

  def testForwardND(self):
    self._test_forward(size=(2, 3, 4))

  def testBackward(self):
    self._test_backward(size=())

  def testBackwardND(self):
    self._test_backward(size=(2, 3, 4))


class TestCat(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    ctx = [None]
    a = torch.rand(size=(3, 4), dtype=self.dtype)
    b = torch.rand(size=(5, 4), dtype=self.dtype)
    y = cat([a, b], dim=0, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(a, backward_args[1], "a isn't in backward_args")
    assert_tensor_in(b, backward_args[1], "b isn't in backward_args")

  def _test_forward(self, sizes, dim):
    tensors = [
      torch.rand(size=sizes[i], dtype=self.dtype)
      for i in range(len(sizes))
    ]
    y = cat(tensors, dim=dim)
    y_ = torch.cat(tensors, dim=dim)
    dbg = f'tensors: {tensors}\ndim: {dim}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, sizes, dim):
    tensors = [
      torch.rand(size=sizes[i], dtype=self.dtype)
      for i in range(len(sizes))
    ]
    ctx = []
    for t in tensors:
      t.grad = torch.zeros_like(t)
    y = cat(tensors, dim=dim, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      tensors_ = [t.detach().clone().requires_grad_() for t in tensors]
      y_ = torch.cat(tensors_, dim=dim)
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)
    torch.autograd.backward([y_], [y.grad])

    dbg = f'tensors: {tensors}\ndim: {dim}'
    for i, (t, t_) in enumerate(zip(tensors, tensors_)):
      dbg_t = dbg + f'\ngot (tensors[{i}].grad): {t.grad}\nexpected (tensors[{i}].grad): {t_.grad}'
      torch.testing.assert_allclose(t.grad, t_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_t)

  def testForward_0_123_223(self):
    self._test_forward(sizes=((1, 2, 3), (2, 2, 3)), dim=0)

  def testForward_1_213_223(self):
    self._test_forward(sizes=((2, 1, 3), (2, 2, 3)), dim=1)

  def testForward__1_123_124(self):
    self._test_forward(sizes=((1, 2, 3), (1, 2, 4)), dim=-1)

  def testBackward_0_123_223(self):
    self._test_backward(sizes=((1, 2, 3), (2, 2, 3)), dim=0)

  def testBackward_1_213_223(self):
    self._test_backward(sizes=((2, 1, 3), (2, 2, 3)), dim=1)

  def testBackward__1_123_124(self):
    self._test_backward(sizes=((1, 2, 3), (1, 2, 4)), dim=-1)


class TestUnbind(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    ctx = [None]
    x = torch.rand(size=(3, 4, 5), dtype=self.dtype)
    y_tensors = unbind(x, dim=0, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    for i, yi in enumerate(y_tensors):
      assert_tensor_in(yi, backward_args[0], f"y_tensors[{i}] isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")

  def _test_forward(self, size, dim):
    x = torch.rand(size=size, dtype=self.dtype)
    y_tensors = unbind(x, dim=dim)
    y_tensors_ = torch.unbind(x, dim=dim)
    assert len(y_tensors) == len(
      y_tensors_
    ), f"incorrect length. got: {len(y_tensors)}. expected: {len(y_tensors_)}"
    for i, (yi, yi_) in enumerate(zip(y_tensors, y_tensors_)):
      assert yi.shape == yi_.shape, f"incorrect shape in index {i}. got: {yi.shape}. expected: {yi_.shape}"
      dbg = f'x: {x}\ndim: {dim}\ngot ({i}): {yi}\nexpected ({i}): {yi_}'
      torch.testing.assert_allclose(yi, yi_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size, dim):
    x = torch.rand(size=size, dtype=self.dtype)
    ctx = []
    x.grad = torch.zeros_like(x)
    y_tensors = unbind(x, dim=dim, ctx=ctx)
    for y in y_tensors:
      y.grad = torch.zeros_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone().requires_grad_()
      y_tensors_ = torch.unbind(x_, dim=dim)
    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)
    torch.autograd.backward(y_tensors_, [y.grad for y in y_tensors])

    dbg = f'x: {x}\ndim: {dim}\ngrad: {[y.grad for y in y_tensors]}\ngot (x.grad): {x.grad}\nexpected (x.grad): {x_.grad}'
    torch.testing.assert_allclose(x.grad, x_.grad, rtol=self.rtol, atol=self.atol, msg=dbg)

  def testForward_0_234(self):
    self._test_forward(size=(2, 3, 4), dim=0)

  def testForward_1_234(self):
    self._test_forward(size=(2, 3, 4), dim=1)

  def testForward__1_234(self):
    self._test_forward(size=(2, 3, 4), dim=-1)

  def testBackward_0_234(self):
    self._test_backward(size=(2, 3, 4), dim=0)

  def testBackward_1_234(self):
    self._test_backward(size=(2, 3, 4), dim=1)

  def testBackward__1_234(self):
    self._test_backward(size=(2, 3, 4), dim=-1)


class TestEmbedding(unittest.TestCase):
  def setUp(self):
    self.atol = 1e-6
    self.rtol = 1e-6
    self.dtype = torch.float32

  def testContext(self):
    size_x = (3, 4)
    size_w = (10, 5)
    ctx = [None]
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    w = torch.rand(size=size_w, dtype=self.dtype)
    y = embedding(x, w, ctx=ctx)
    assert len(ctx) == 2, "didn't add new call to context"
    assert ctx[0] is None, "modified existing context"
    backward_fn, backward_args = ctx[1]
    # assert backward_fn == view_backward, "wrong backward_fn"
    assert_tensor_in(y, backward_args, "y isn't in backward_args")
    assert_tensor_in(x, backward_args, "x isn't in backward_args")
    assert_tensor_in(w, backward_args, "w isn't in backward_args")

  def _test_forward(self, size_x, size_w):
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    w = torch.rand(size=size_w, dtype=self.dtype)
    y = embedding(x, w)
    y_ = torch.nn.functional.embedding(x,w)
    dbg = f'x: {x}\nw: {w}\ngot: {y}\nexpected: {y_}'
    assert y.shape == y_.shape, f"incorrect shape. got: {y.shape}. expected: {y_.shape}"
    torch.testing.assert_allclose(y, y_, rtol=self.rtol, atol=self.atol, msg=dbg)

  def _test_backward(self, size_x, size_w):
    x = torch.randint(low=0, high=size_w[0], size=size_x, dtype=torch.long)
    w = torch.rand(size=size_w, dtype=self.dtype)
    ctx = []
    w.grad = torch.zeros_like(w)
    y = embedding(x, w, ctx=ctx)
    y.grad = torch.rand_like(y)

    with torch.enable_grad():
      x_ = x.detach().clone()
      w_ = w.detach().clone().requires_grad_()
      y_ = torch.nn.functional.embedding(x_,w_)

    backward_fn, backward_args = ctx.pop()
    backward_fn(*backward_args)

    torch.autograd.backward([y_], [y.grad])

    dbg = f'x: {x}\nw: {w}\ny.grad: {y.grad}'
    dbg_w = dbg + f'\ngot (w.grad): {w.grad}\nexpected (w.grad): {w_.grad}'
    torch.testing.assert_allclose(w.grad, w_.grad, rtol=self.rtol, atol=self.atol, msg=dbg_w)

  def testForward(self):
    self._test_forward(size_x=(3, 4), size_w=(10, 5))

  def testBackward(self):
    self._test_backward(size_x=(3, 4), size_w=(10, 5))

