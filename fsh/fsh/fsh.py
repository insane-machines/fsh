from math import nan, sqrt
import numpy as np

class Linear():
    def __init__(self, n_features=1, lr = 0.001):
        self.lr         = lr
        self.weight     = np.random.randn(n_features, 1)
        self.bias       = 1
        self.n_features = n_features
        self.stop = False
        self.history = {'epoch': [], 'loss': [], 'val_loss': []}
    def predict(self, x, training=False):
        y_pred = x @ self.weight + self.bias
        if not training:
            print(f'▶ Predicted output for x = {x}: {y_pred}')
        return y_pred

    def calc_diff(self, param_name, x, y, d_h=1e-5, training=True):
        orig_num = getattr(self, param_name)
        
        setattr(self, param_name, orig_num+d_h)
        loss_p = metrics.mse(y, self.predict(x, training), training)
        
        setattr(self, param_name, orig_num-d_h)
        loss_m = metrics.mse(y, self.predict(x, training), training)
        
        setattr(self, param_name, orig_num)

        grad = (loss_p - loss_m) / (2 * d_h)
        return grad

    def calc_grad(self, x, y, training=True):
        error = y - self.predict(x, training=training)
        n = x.shape[0]
        grad_w = -2 / n * (x.T @ error)
        grad_b = -2 / n * np.sum(error)

        return grad_w, grad_b

    def train(self, x, y, val_x=None, val_y=None, epochs=50, loss_fn=None, training=True, callbacks=None, view_epoch=1, batch_size=1, \
                                                                                    diff_height=1e-5):
        x = preprocessing.to_array(x)
        y = preprocessing.to_array(y)

        if val_x is not None and val_y is not None:
            val_x = preprocessing.to_array(val_x)
            val_y = preprocessing.to_array(val_y)
        
        if loss_fn is None:
            loss_fn = metrics.mse
        #MAIN CYCLE
        for epoch in range(1, epochs+1):
            if type(x) != np.ndarray or type(y) != np.ndarray:
                raise DataError(f'TypeError: Invalid type of training data: needs <np.ndarray>')

            if len(x) > len(y) or len(y) > len(x):
                raise MatchError('Sizes of datas is not match')

            x_data = np.split(x, np.arange(batch_size, len(x), batch_size))
            y_data = np.split(y, np.arange(batch_size, len(y), batch_size))
            for batch in range(len(x_data)):
                x_batch = x_data[batch]
                y_batch = y_data[batch]

                grad_w, grad_b = self.calc_grad(x_batch, y_batch, training=training)
            
                self.weight -= self.lr * grad_w
                self.bias   -= self.lr * grad_b

                y_pred = self.predict(x_batch, training)
                loss = loss_fn(y_batch, y_pred, training)
                if loss is None:
                    raise ProcessError(f'detected None on epoch {epoch}, batch {batch}')

                if val_x is None and val_y is None:
                    valprint = ""
                    val_loss = None
                else:
                    val_pred = self.predict(val_x, training)
                    val_loss = loss_fn(val_y, val_pred, training)
                    valprint = f', validation_loss: {val_loss}'

                if epoch % view_epoch == 0:
                    total_batches = (len(x) + batch_size - 1) // batch_size
                    batch_per = (batch + 1) / total_batches 
                    per = int(batch_per * 20)
                    per = min(per, 20)  

                    print("▮" * per + "▯" * (20 - per), f"{batch_per*len(x)}/{len(x)} Epoch {epoch}, loss: {loss}" + valprint)

            logs = {'loss': loss, 'val_loss': val_loss}
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)
            self.history['epoch'].append(epoch)
            if callbacks is not None:
                for cb in callbacks:
                    cb(self, logs)
            
            if self.stop:
                print(f'Early stopped at the epoch {epoch}')
                break

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

class callbacks():
    class EarlyStop():
        def __init__(self, patience=10, delta=1e-3):
            self.best_loss = None
            self.delta = delta
            self.patience = patience
            self.wait = 0
        def __call__(self, model, log):
            val_loss = log.get('val_loss')
            if val_loss is None:
                return

            if self.best_loss == None:
                self.best_loss = val_loss
            
            else:
                if val_loss < self.best_loss - self.delta:
                    self.best_loss = val_loss
                    self.wait = 0
                else:
                    self.wait += 1

            if self.wait >= self.patience:
                model.stop = True
            else:
                model.stop = False

class preprocessing():
    @staticmethod
    def normalize(array):
        arr_mean    = np.mean(array)
        arr_std     = np.std(array)
        arr_norm    = (array-arr_mean)/arr_std
        return arr_norm, arr_mean, arr_std
    @staticmethod
    def denormalize(normalized, std, mean):
        array = normalized * std + mean
        return array
    @staticmethod
    def to_array(dataframe):
        array = np.asarray(dataframe)
        return array

class MatchError(Exception):
    pass
class DataError(Exception):
    pass
class ProcessError(Exception):
    pass
        