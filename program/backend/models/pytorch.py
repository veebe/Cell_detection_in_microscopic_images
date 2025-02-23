import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from backend.backend_types import modelTypes
from backend.models.model import BaseModel
from backend.training.trainingThread import TrainingThreadPyTorch

class PyTorchModel(BaseModel, nn.Module):
    def __init__(self, model_type=modelTypes.UNET, backbone="resnet34", input_channels=3, num_classes=1, device='cuda', input_size=(256, 256)):
        super(PyTorchModel, self).__init__()
        nn.Module.__init__(self) 

        if device == "cuda" and torch.cuda.is_available():
            self.device = device
        else:
            self.device = "cpu"

        print(self.device)
        print(model_type)
        print(backbone)
        print(input_size)

        if model_type == modelTypes.UNET:
            self.model = smp.Unet(encoder_name=backbone, in_channels=input_channels, classes=num_classes)
        elif model_type == modelTypes.UNETPP:
            self.model = smp.UnetPlusPlus(encoder_name=backbone, in_channels=input_channels, classes=num_classes)
        elif model_type == modelTypes.DEEPLABV3:
            self.model = smp.DeepLabV3(encoder_name=backbone, in_channels=input_channels, classes=num_classes)
        elif model_type == modelTypes.FNP:
            self.model = smp.FPN(encoder_name=backbone, in_channels=input_channels, classes=num_classes)

        self.model.to(self.device)

        self.optimizer = None
        self.criterion = None
        self.epochs = 10

        self.model_type = model_type
        self.backbone = backbone
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.device = device
        self.input_size = input_size

        self.metrics = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    def forward(self, x):
        return self.model(x)

    def compile(self, optimizer="adam", loss="bce"):
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        
        self.criterion = nn.BCEWithLogitsLoss() if loss == "bce" else nn.CrossEntropyLoss()

    def _train(self, train_loader: DataLoader, val_loader: DataLoader, callbacks=None):
        if not self.optimizer or not self.criterion:
            raise ValueError("Model must be compiled before training.")

        self.thread = TrainingThreadPyTorch(
            model=self,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.epochs,
            callbacks=callbacks,
            device = self.device
        )
        self.thread.start()

    def train_epoch(self, train_loader, callback):
        self.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.criterion(output, target.float())
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            preds = torch.sigmoid(output) > 0.5
            correct += (preds == target).sum().item()
            total += target.numel()
            
        accuracy = correct / total
        return epoch_loss / len(train_loader), accuracy

    def validate(self, val_loader):
        self.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self(data)
                val_loss += self.criterion(output, target.float()).item()
                
                preds = torch.sigmoid(output) > 0.5
                correct += (preds == target).sum().item()
                total += target.numel()
                
        accuracy = correct / total
        return val_loss / len(val_loader), accuracy

    def predict(self, inputs: torch.Tensor):
        self.eval()
        with torch.no_grad():
            return self(inputs)

    def save(self, path: str):
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'model_type': self.model_type,
            'backbone': self.backbone,
            'input_channels': self.input_channels,
            'num_classes': self.num_classes,
            'device': self.device,
            'input_size': self.input_size
        }, path)

    @classmethod
    def load(cls, path: str):
        checkpoint = torch.load(path)
        model = cls(
            model_type=checkpoint['model_type'],
            backbone=checkpoint['backbone'],
            input_channels=checkpoint['input_channels'],
            num_classes=checkpoint['num_classes'],
            device=checkpoint['device'],
            input_size=checkpoint['input_size']
        )
        model.model.load_state_dict(checkpoint['model_state'])
        model.compile()
        if 'optimizer_state' in checkpoint:
            model.optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model
    
    def save_weights(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        print("Model weights loaded successfully.")
