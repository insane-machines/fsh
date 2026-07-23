from math import sqrt
from ..backend.backend_manager import BackendManager
class metrics():

    @staticmethod
    def mse(y, y_pred, training=False):
        backend = BackendManager.get_backend()
        y       = backend.convert(y)
        y_pred  = backend.convert(y_pred)
        
        loss = backend.mean((y.flatten()-y_pred.flatten())**2)
        if not training:
            print(f'MSE: {loss}')
        return loss

    @staticmethod
    def rmse(y, y_pred, training=False):
        y       = BackendManager.get_backend().convert(y)
        y_pred  = BackendManager.get_backend().convert(y_pred)

        loss = sqrt(metrics.mse(y, y_pred, training=True))
        if not training:
            print(f'RMSE: {loss}')
        return loss

    @staticmethod
    def mae(y, y_pred, training=False):
        backend = BackendManager.get_backend()
        mae = backend.mean(backend.abs(y - y_pred))

        if not training:
            print('MAE:', mae)

        return mae