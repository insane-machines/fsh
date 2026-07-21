from autograd.operation import Operation
from tensor import Tensor

class Div(Operation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        self.parents.append(a)
        self.parents.append(b)
        
        result = Tensor(a.data - b.data, requires_grad=a.requires_grad or b.requires_grad)
        result.grad_fn = self # type: ignore

        return result 

    
    def backward(self, result_grad):
        a = self.parents[0]
        b = self.parents[1]

        if a.requires_grad:
            a.grad += result_grad
            
            if a.grad_fn is not None:
                a.grad_fn.backward(a.grad)  

        if b.requires_grad:
            b.grad -= result_grad
            
            if b.grad_fn is not None:
                b.grad_fn.backward(b.grad)  
        