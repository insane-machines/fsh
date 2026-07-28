from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train():
        pass

    