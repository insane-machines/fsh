from abc import ABC, abstractmethod
from typing import Optional
from ..training.layers.layer import Layer

class Optimizer(ABC):
    def __init__(self, learning_rate=1e-3) -> None:
        super().__init__()
        self.model: Optional[Layer] = None
        self.params = []
        self.learning_rate = learning_rate

    def get_params(self, model: Layer):
        self.model = model
        params = self.model.__dict__ 

        for param in params.values():
            if param.requires_grad:
                self.params.append(param)
            
    
    @abstractmethod
    def step():
        pass
