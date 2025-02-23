import json
import segmentation_models as sm
import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
from backend.models.model import BaseModel
from backend.training.trainingThread import TrainingThreadKeras

class KerasModel(BaseModel):
    def __init__(self, backbone="resnet34", input_size=(256, 256, 3), num_classes=1):
        self.backbone = backbone
        self.input_size = input_size
        self.num_classes = num_classes
        self.epochs = 10

        self.model = sm.Unet(
            backbone, 
            input_shape=input_size, 
            encoder_weights="imagenet", 
            classes=num_classes, 
            activation="sigmoid"
        )
        
        self.model._metadata = {
            "backbone": backbone,
            "input_size": input_size,
            "num_classes": num_classes,
            "epochs": self.epochs
        }

    def compile(self, optimizer="adam", loss="binary_crossentropy"):
        opt = Adam() if optimizer == "adam" else tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    def _train(self, train_data, val_data, batch_size=16, callbacks=None):
        self.thread = TrainingThreadKeras(
            model = self.model,
            training_data = train_data, 
            validation_data = val_data,
            epochs = self.epochs,
            callbacks = callbacks
        )
        self.thread.start()

    def predict(self, inputs):
        return self.model.predict(inputs)

    def save(self, path: str):
        self.model.save(path)
        metadata_path = f"{path}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model._metadata, f)

    @classmethod
    def load(cls, path: str, meta_path : str = ""):
        model = tf.keras.models.load_model(path)
        
        if meta_path == "":
            metadata_path = f"{path}_metadata.json"
        else:
            metadata_path = meta_path
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError("Metadata file missing. Save with KerasModel.save()")

        instance = cls(
            backbone=metadata["backbone"],
            input_size=tuple(metadata["input_size"]),
            num_classes=metadata["num_classes"]
        )
        instance.epochs = metadata["epochs"]
        instance.model = model
        instance.compile()
        return instance

    def save_weights(self, path: str):
        self.model.save_weights(path)
        metadata = {
            "backbone": self.backbone,
            "input_size": self.input_size,
            "num_classes": self.num_classes
        }
        with open(f"{path}_metadata.json", 'w') as f:
            json.dump(metadata, f)

    def load_weights(self, path: str):
        try:
            with open(f"{path}_metadata.json", 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            raise ValueError("Weight metadata missing. Use save_weights() to preserve metadata")

        if (metadata["backbone"] != self.backbone or
            metadata["input_size"] != list(self.input_size) or
            metadata["num_classes"] != self.num_classes):
            raise ValueError("Model architecture doesn't match weight metadata")
        
        self.model.load_weights(path)