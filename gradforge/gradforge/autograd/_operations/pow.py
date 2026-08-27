from ...autograd.operation import Operation
class Pow(Operation):
    def __init__(self, backend) -> None:
        super().__init__(backend)
        self.n = 0
        self.parent_data = 0

    def forward(self, x, n):

        self.parent_data = x
        self.n = n

        result = x ** n

        return result 

    
    def backward(self, result_grad) -> list:
        x_data = self.parent_data
        n = self.n
        
        x_grad = result_grad * n * x_data**(n - 1)

        return [x_grad]

        