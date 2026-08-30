from ...autograd.operation import Operation

class Div(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.parent_data = (0, 0)

    def forward(self, a, b):
        
        result = self._backend.div(a, b)
        self.parent_data = (a, b)

        return result 

    
    def backward(self, result_grad) -> tuple:
        a_data, b_data = self.parent_data

        a_grad = result_grad / b_data if b_data != 0 else 0 
        b_grad = (- a_data / b_data ^ 2) * result_grad if b_data != 0 else 0 

        return (a_grad, b_grad)
        