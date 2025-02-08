from PyQt5.QtCore import QThread

class TrainingThread(QThread):
    def __init__(self, model, training_data, validation_data, epochs, callback):
        super().__init__()
        self.model = model
        self.training_data = training_data  
        self.validation_data = validation_data  
        self.epochs = epochs
        self.callback = callback

    def run(self):
        try:
            history = self.model.fit(
                self.training_data,
                epochs=self.epochs,
                validation_data=self.validation_data,
                callbacks=[self.callback],
                verbose=0,
                shuffle=False 
            )
            
            self.callback.history = history.history
            
        except Exception as e:
            self.callback.training_failed.emit(str(e))
            raise e
        finally:
            if hasattr(self.training_data, 'close'):
                self.training_data.close()