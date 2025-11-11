from fsh.additional.preprocess import preprocessing
import numpy as np
from math import sqrt

class metrics():
    @staticmethod
    def mse(y, y_pred, training=False):
        y       = preprocessing.to_array(y)
        y_pred  = preprocessing.to_array(y_pred)
        
        loss = np.mean((y.flatten()-y_pred.flatten())**2)
        if not training:
            print(f'MSE: {loss}')
        return loss

    @staticmethod
    def rmse(y, y_pred, training=False):
        y       = preprocessing.to_array(y)
        y_pred  = preprocessing.to_array(y_pred)

        loss = sqrt(metrics.mse(y, y_pred, training=True))
        if not training:
            print(f'RMSE: {loss}')
        return loss

    @staticmethod
    def mae(y, y_pred, training=False):
        mae = np.mean(np.abs(y - y_pred))

        if not training:
            print('MAE:', mae)

        return mae