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
    class History():
        def __init__(self) -> None:
            self.history = {'epoch': [], 'loss': [], 'val_loss': []}
        def __call__(self, model, log):
            epoch = log.get('epoch')
            loss = log.get('loss')
            val_loss = log.get('val_loss')
            self.history['epoch'].append(epoch)
            self.history['loss'].append(loss)
            self.history['val_loss'].append(val_loss)