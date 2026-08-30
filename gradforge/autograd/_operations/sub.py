from ...autograd.operation import Operation
class Sub(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def forward(self, a, b):
        
        result = self._backend.sub(a.data, b.data)

        return result 

    
    def backward(self, result_grad) -> tuple:

        a_grad = result_grad
        b_grad = -result_grad

        return (a_grad, b_grad)
        