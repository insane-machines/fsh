from .backend import Backend
from typing import Optional
class BackendManager():
    _backend: Optional[Backend] = None

    @staticmethod
    def set_backend(backend_adapter: Backend):
        if backend_adapter is None:
            raise ValueError("Backend adapter cannot be None")
        
        BackendManager._backend = backend_adapter

    @staticmethod
    def get_backend():
        if BackendManager._backend is not None:
            return BackendManager._backend
        else: 
            raise RuntimeError("Backend is not initialized yet. \
                               Please set backend with BackendManager.set(<your backend>)")
        