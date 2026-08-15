from ...autograd.operation import Operation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.tensor import Tensor
class Pow(Operation):
    def __init__(self) -> None:
        super().__init__()
        self.n = 0

    def forward(self, x: "Tensor", n) -> "Tensor":
        from ...core.tensor import Tensor
        self._parents.append(x)
        self.n = n

        result = Tensor(x.data ** n, requires_grad=x.requires_grad)
        result.grad_fn = self 

        return result 

    
    def backward(self, result_grad) -> list:
        x: "Tensor" = self._parents[0]
        n = self.n
        
        x.grad += result_grad * n * x.data**(n - 1)

        return [(x, x.grad)]

        