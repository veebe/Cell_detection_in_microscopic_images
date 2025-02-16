import segmentation_models as sm
import tensorflow as tf
from keras._tf_keras.keras.optimizers import Adam
#from tensorflow.keras.optimizers import Adam
from backend.models.model import BaseModel
from backend.training.trainingThread import TrainingThreadKeras

class KerasModel(BaseModel):
    def __init__(self, backbone="resnet34", input_size=(256, 256, 3), num_classes=1):
        backbone = backbone.lower()
        self.model = sm.Unet(backbone, input_shape=input_size, encoder_weights="imagenet", classes=num_classes, activation="sigmoid")
        self.epochs = 10

    def compile(self, optimizer="adam", loss="binary_crossentropy"):

        opt = Adam() if optimizer == "adam" else tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])

    def _train(self, train_data, val_data, batch_size=16, callbacks=None):
        #self.model.fit(train_data, validation_data=val_data, epochs=self.epochs, batch_size=batch_size)
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

    def load(self, path: str):
        self.model = tf.keras.models.load_model(path)

    def save_weights(self, path: str):
        self.model.save_weights(path)
