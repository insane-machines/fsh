import numpy as np
from fsh.errors.errors import DataError
class preprocessing():
    @staticmethod
    def normalize(array, mean=None, std=None):
        if mean is not None and std is not None:
            arr_mean = mean
            arr_std = std
        elif (mean is None and std is not None) or (mean is not None and std is None):
            raise DataError('one of the normalize components is none, and another is not none')
        else:
            arr_mean    = np.mean(array)
            arr_std     = np.std(array)
        arr_norm    = (array-arr_mean)/arr_std
        return arr_norm, arr_mean, arr_std
    @staticmethod
    def denormalize(normalized, mean, std):
        array = normalized * std + mean
        return array
    @staticmethod
    def to_array(dataframe):
        array = np.asarray(dataframe)
        return array