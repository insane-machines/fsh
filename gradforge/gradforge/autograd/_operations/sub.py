from ...autograd.operation import Operation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.tensor import Tensor
class Sub(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def forward(self, a: "Tensor", b: "Tensor") -> Tensor:
        from ...core.tensor import Tensor
        self._parents.append(a)
        self._parents.append(b)
        
        result = Tensor(self._backend.sub(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
        result.grad_fn = self 

        return result 

    
    def backward(self, result_grad) -> list:
        a: "Tensor" = self._parents[0]
        b: "Tensor" = self._parents[1]

        if a.requires_grad:
            a.grad += result_grad 

        if b.requires_grad:
            b.grad -= result_grad

        return [(a, a.grad), (b, b.grad)]
        