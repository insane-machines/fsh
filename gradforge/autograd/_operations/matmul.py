from ...autograd.operation import Operation
class Matmul(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.parent_data = (0, 0)

    def forward(self, a, b):

        self.parent_data = (a, b)
        result = self._backend.matmul(a, b)
    
        return result 

    
    def backward(self, result_grad) -> tuple:

        a_data, b_data = self.parent_data
        
        a_grad = result_grad @ self._backend.transpose(b_data) 
        b_grad = self._backend.transpose(a_data) @ result_grad 

        return (a_grad, b_grad)