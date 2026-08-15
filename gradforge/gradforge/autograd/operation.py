from __future__ import annotations
from abc import ABC, abstractmethod
from ..backend.backend import Backend
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.tensor import Tensor
from ..backend.backend_manager import BackendManager
class Operation(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._parents: list["Tensor"] = []
        self._backend: Backend = BackendManager.get_backend() 

    @abstractmethod
    def forward(self, a, b) -> "Tensor":
        pass 

    @abstractmethod
    def backward(self, result_grad) -> list:
        pass 
