from ...autograd.operation import Operation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.tensor import Tensor

class Div(Operation):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: "Tensor", b: "Tensor") -> "Tensor":
        from ...core.tensor import Tensor

        self._parents.append(a)
        self._parents.append(b)
        
        result = Tensor(self._backend.div(a, b), requires_grad=a.requires_grad or b.requires_grad)
        result.grad_fn = self # type: ignore

        return result 

    
    def backward(self, result_grad) -> list:
        a: "Tensor" = self._parents[0]
        b: "Tensor" = self._parents[1]

        if a.requires_grad:
            a.grad += result_grad / b.data if b.data != 0 else 0 

        if b.requires_grad:
            b.grad += (- a.data / b.data ^ 2) * result_grad if b.data != 0 else 0 

        return [(a, a.grad), (b, b.grad)]
        