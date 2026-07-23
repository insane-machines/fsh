from .backend import Backend
from .numpy_backend.numpy_backend import NumpyBackend
class BackendManager():
    _backend: Backend = NumpyBackend()

    @staticmethod
    def set_backend(backend_adapter: Backend):
        BackendManager._backend = backend_adapter

    @staticmethod
    def get_backend() -> Backend:
        return BackendManager._backend
        