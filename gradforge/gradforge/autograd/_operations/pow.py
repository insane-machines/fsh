from autograd.operation import Operation
from tensor import Tensor

class Pow(Operation):
    def __init__(self) -> None:
        self.parent: Tensor = None # type: ignore
        self.n = 0

    def forward(self, x: Tensor, n) -> Tensor:
        self.parent = x
        self.n = n

        result = Tensor(x.data^n, requires_grad=x.requires_grad)
        result.grad_fn = self # type: ignore

        return result 

    
    def backward(self, result_grad):
        x = self.parent
        n = self.n
        
        x.grad += result_grad * n * x^(n - 1)
        x.grad_fn.backward(x.grad)

        