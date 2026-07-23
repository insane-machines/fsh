from __future__ import annotations
from ..backend.backend_manager import BackendManager
from ..backend.backend import Backend
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..autograd.operation import Operation
from ..autograd._operations import mul, matmul, sub, div, add
from typing import Optional

class Tensor():

    def __init__(self, data, requires_grad: bool = False):
        self.requires_grad: bool = requires_grad
        self.grad = 0
        self.grad_fn: Optional["Operation"] = None 
        self.backend: Backend = BackendManager.get_backend()
        self.data = self.backend.convert(data)

    def __mul__(self, other): 
        return mul.Mul().forward(self, other)

    def __matmul__(self, other):
        return matmul.Matmul().forward(self, other)

    def __add__(self, other):
        return add.Add().forward(self, other)

    def __sub__(self, other):
        return sub.Sub().forward(self, other)

    def __truediv__(self, other):
        return div.Div().forward(self, other)

    def backward(self):
        start_grad = self.backend.ones_like(self.data)
        if self.grad_fn:
            self.grad_fn.backward(start_grad)
        else:
            raise RuntimeError("Backward function was called but not set")