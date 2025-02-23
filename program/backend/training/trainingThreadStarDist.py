from PyQt5.QtCore import QThread

class TrainingThreadStarDist(QThread):
    def __init__(self, model, X_train, Y_train, X_val, Y_val, callbacks):
        super().__init__()
        self.model = model
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.callbacks = callbacks

    def run(self):
        try:
            self.model._train(
                self.X_train, 
                self.Y_train,
                self.X_val,
                self.Y_val,
                callbacks=self.callbacks
            )
            self.callbacks.training_completed.emit()
        except Exception as e:
            #self.callbacks.training_failed.emit(str(e))
            raise e