from ...autograd.operation import Operation

class Add(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)

    def forward(self, a, b):
        
        result = self._backend.add(a, b)
        
        return result 

    
    def backward(self, result_grad) -> tuple:

        a_grad = result_grad
        b_grad = result_grad

        return (a_grad, b_grad)
        