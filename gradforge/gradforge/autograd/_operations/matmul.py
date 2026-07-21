from autograd.operation import Operation
from tensor import Tensor

class MatmulOperation(Operation):
    def __init__(self) -> None:
        self.parents: list[Tensor] = []

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.parents.append(a)
        self.parents.append(b)
        
        result = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.grad_fn = self # type: ignore

        return result 

    
    def backward(self, result_grad):
        a = self.parents[0]
        b = self.parents[1]

        if a.requires_grad:
            a.grad += b.data.T @ result_grad
            a.grad_fn.backward(a.grad)  

        if b.requires_grad:
            b.grad += a.data.T @ result_grad
            b.grad_fn.backward(b.grad)
        