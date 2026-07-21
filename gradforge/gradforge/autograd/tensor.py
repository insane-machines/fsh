from backend.backend_manager import BackendManager
from tensor import Tensor
from operation import Operation
from _operations import mul, matmul, sub, div, add

class Tensor():

    def __init__(self, data, requires_grad: bool = False):
        self.data = data
        self.requires_grad: bool = requires_grad
        self.grad = 0
        self.grad_fn: Operation = None # type: ignore
        self.backend = BackendManager._backend

    def __mul__(self, other: Tensor):
        return mul.MulOperation().forward(self, other)

    def __matmul__(self, other: Tensor):
        return matmul.MatmulOperation().forward(self, other)

    def __add__(self, other: Tensor):
        return add.AddOperation().forward(self, other)

    def __sub__(self, other: Tensor):
        return sub.SubOperation().forward(self, other)

    def __truediv__(self, other: Tensor):
        return div.DivOperation().forward(self, other)
