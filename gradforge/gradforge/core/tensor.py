from __future__ import annotations
from ..backend.backend_manager import BackendManager
from ..backend.backend import Backend
from ..autograd._operations import mul, matmul, sub, div, add, pow
from typing import Optional

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..autograd.operation import Operation

class Tensor():

    def __init__(self, data, requires_grad: bool = False):
        self.requires_grad: bool = requires_grad
        self.grad = 0
        self.grad_fn: Optional["Operation"] = None 
        self.backend: Backend = BackendManager.get_backend()
        self.data = self.backend.convert(data)

    def __mul__(self, other): 
        return mul.Mul(self.backend).forward(self, other)

    def __matmul__(self, other):
        return matmul.Matmul(self.backend).forward(self, other)

    def __add__(self, other):
        return add.Add(self.backend).forward(self, other)

    def __sub__(self, other):
        return sub.Sub(self.backend).forward(self, other)

    def __truediv__(self, other):
        return div.Div(self.backend).forward(self, other)

    def __pow__(self, other):
        return pow.Pow(self.backend).forward(self, other)
        