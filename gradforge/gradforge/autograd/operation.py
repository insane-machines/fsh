from abc import ABC, abstractmethod
from backend.backend import Backend
from core.tensor import Tensor
from backend.backend_manager import BackendManager
from typing import Optional
class Operation(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._parents: list[Tensor] = []
        self._backend: Optional[Backend] = BackendManager.get_backend() 

    @abstractmethod
    def forward(self, a, b) -> Tensor:
        pass 

    @abstractmethod
    def backward(self, result_grad):
        pass 
