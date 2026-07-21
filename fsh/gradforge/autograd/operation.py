from abc import ABC, abstractmethod
from tensor import Tensor

class Operation(ABC):
    
    @abstractmethod
    def forward(self, a, b) -> Tensor:
        pass 

    @abstractmethod
    def backward(self, result_grad):
        pass 
