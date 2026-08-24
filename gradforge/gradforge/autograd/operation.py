from __future__ import annotations
from abc import ABC, abstractmethod
from ..backend.backend import Backend
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core.tensor import Tensor
class Operation(ABC):
    def __init__(self, backend: Backend) -> None:
        super().__init__()
        self._parents: list["Tensor"] = []
        self._backend: Backend = backend

    @abstractmethod
    def forward(self, a, b) -> "Tensor":
        pass 

    @abstractmethod
    def backward(self, result_grad) -> list:
        pass 
