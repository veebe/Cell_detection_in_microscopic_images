from PyQt5.QtCore import QThread
import torch

class TrainingThreadKeras(QThread):
    def __init__(self, model, training_data, validation_data, epochs, callbacks):
        super().__init__()
        self.model = model
        self.training_data = training_data  
        self.validation_data = validation_data  
        self.epochs = epochs
        self.callbacks = callbacks

    def run(self):
        try:
            from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
            callbacks_list = [
                self.callbacks,
                ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=3, 
                    min_lr=0.00001
                )
            ]

            history = self.model.fit(
                self.training_data,
                epochs=self.epochs,
                validation_data=self.validation_data,
                callbacks=callbacks_list,
                verbose=0,
                shuffle=False 
            )
            
            self.callbacks.history = history.history
            
        except Exception as e:
            self.callbacks.training_failed.emit(str(e))
            raise e
        finally:
            if hasattr(self.training_data, 'close'):
                self.training_data.close()

class TrainingThreadPyTorch(QThread):
    def __init__(self, model, train_loader, val_loader, epochs, callbacks, device):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.callbacks = callbacks
        self.device = device

    def run(self):
        try:
            for epoch in range(self.epochs):
                train_loss, train_acc = self._train_epoch()
                
                val_loss, val_acc = self._validate()
                
                self._emit_metrics(epoch, train_loss, train_acc, val_loss, val_acc)

            self.callbacks.training_completed.emit()
            
        except Exception as e:
            self.callbacks.training_failed.emit(str(e))

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for data, target in self.train_loader:
            data = data.to(self.device)
            target = target.to(self.device)
            
            self.model.optimizer.zero_grad()
            output = self.model(data)
            loss = self.model.criterion(output, target)
            loss.backward()
            self.model.optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.sigmoid(output) > 0.5
            correct += (preds == target).sum().item()
            total += target.numel()
            
        return epoch_loss / len(self.train_loader), correct / total

    def _validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)
                
                output = self.model(data)
                val_loss += self.model.criterion(output, target).item()
                preds = torch.sigmoid(output) > 0.5
                correct += (preds == target).sum().item()
                total += target.numel()
                
        return val_loss / len(self.val_loader), correct / total

    def _emit_metrics(self, epoch, train_loss, train_acc, val_loss, val_acc):
        metrics = {
            'epoch': epoch + 1,
            'loss': f"{train_loss:.4f}",
            'accuracy': f"{train_acc:.4f}",
            'val_loss': f"{val_loss:.4f}",
            'val_accuracy': f"{val_acc:.4f}"
        }
        
        self.callbacks.progress_updated.emit(epoch + 1)
        self.callbacks.metrics_updated.emit(metrics)