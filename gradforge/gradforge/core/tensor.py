from __future__ import annotations
from ..backend.backend_manager import BackendManager
from ..backend.backend import Backend
from ..autograd._operations import mul, matmul, sub, div, add, pow
from typing import Optional

class Tensor():
    __slots__ = ['data', 'requires_grad', 'grad', 'grad_fn', 'backend']
    
    def __init__(self, data, requires_grad: bool = False):
        self.requires_grad: bool = requires_grad
        self.grad = 0
        self.grad_fn: Optional[list] = None
        self.backend: Backend = BackendManager.get_backend()
        self.data = self.backend.convert(data)

    def __mul__(self, other): 
        mul_op = mul.Mul(self.backend)
        mul_forward = mul_op.forward(self.data, other.data)

        result = Tensor(mul_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self, other], mul_op]

        return result

    def __matmul__(self, other: Tensor):
        mat_mul_op = matmul.Matmul(self.backend)
        mul_forward = mat_mul_op.forward(self.data, other.data)

        result = Tensor(mul_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self, other], mat_mul_op]

        return result

    def __add__(self, other: Tensor):
        add_op = add.Add(self.backend)
        add_forward = add_op.forward(self.data, other.data)

        result = Tensor(add_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self, other], add_op]

        return result

    def __sub__(self, other: Tensor):
        sub_op = sub.Sub(self.backend)
        sub_forward = sub_op.forward(self.data, other.data)

        result = Tensor(sub_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self, other], sub_op]

        return result

    def __truediv__(self, other):
        div_op = div.Div(self.backend)
        div_forward = div_op.forward(self.data, other.data)

        result = Tensor(div_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self, other], div_op]

        return result

    def __pow__(self, other):
        pow_op = pow.Pow(self.backend)
        pow_forward = pow_op.forward(self.data, other)

        result = Tensor(pow_forward, requires_grad=self.requires_grad or other.requires_grad)
        result.grad_fn = [[self], pow_op]

        return result

        