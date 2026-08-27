from __future__ import annotations
from abc import ABC, abstractmethod
from ..backend.backend import Backend
class Operation(ABC):
    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._backend: Backend = backend

    @abstractmethod
    def forward(self, a, b):
        pass 

    @abstractmethod
    def backward(self, result_grad) -> tuple:
        pass 
