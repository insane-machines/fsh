from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, learning_rate=1e-3) -> None:
        super().__init__()
        self.learning_rate = learning_rate
            
    @abstractmethod
    def step():
        pass

    @abstractmethod
    def zero_grad():
        pass
