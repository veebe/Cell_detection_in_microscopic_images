from PyQt5.QtCore import QObject, pyqtSignal
from tensorflow.python.keras.callbacks import Callback

class TrainingProgressCallback(QObject, Callback):
    progress_updated = pyqtSignal(int)  
    metrics_updated = pyqtSignal(dict) 
    training_completed = pyqtSignal()
    training_failed = pyqtSignal(str)

    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        self.progress_updated.emit(epoch + 1)  
        
        metrics = {
            'epoch': epoch + 1,
            'loss': f"{logs.get('loss', 0):.4f}",
            'accuracy': f"{logs.get('accuracy', 0):.4f}",
            'val_loss': f"{logs.get('val_loss', 0):.4f}",
            'val_accuracy': f"{logs.get('val_accuracy', 0):.4f}"
        }
        self.metrics_updated.emit(metrics)
        
    def on_train_end(self, logs=None):
        self.training_completed.emit()