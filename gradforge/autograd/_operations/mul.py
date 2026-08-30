from ...autograd.operation import Operation
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ...core.tensor import Tensor
class Mul(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.parent_data = (0, 0)

    def forward(self, a, b):

        self.parent_data = (a, b)
        result = self._backend.mul(a, b)

        return result 

    
    def backward(self, result_grad) -> tuple:
        a_data, b_data = self.parent_data

        a_grad = b_data * result_grad
        b_grad = a_data * result_grad

        return (a_grad, b_grad)
        