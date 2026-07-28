from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, learning_rate=0.001) -> None:
        super().__init__(learning_rate)

    def step(self):
        for param in self.params:
            param.data -= param.grad * self.learning_rate