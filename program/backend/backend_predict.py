import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
import yolov5 as yol
from PyQt5.QtWidgets import QFileDialog

from enum import Enum

class PredictMethods(Enum):
    UPLOADED_MODEL = 0
    UPLOADED_WEIGHTS = 2
    SELECTED_MODEL = 3 

class PreprocessingSettings:
    def __init__(self):
        self.blur_check = False
        self.blur = 0
        self.brightness_check = False
        self.brightness = 0
        self.contrast_check = False
        self.contrast = 100
        self.denoise_check = False
        self.denoise = 0

class PredictionController:
    def __init__(self):
        self.model_uploaded = None
        self.model_selected = None
        self.weights_uploaded = None
        self.method = ""

        self.eval_image_paths = []
        self.eval_images = []
        self.eval_images_preprocessed = []
        self.eval_images_preprocessed_np = []

        self.preprocess_settings = PreprocessingSettings()
        self.predict_method = PredictMethods.UPLOADED_MODEL
        self.image_width = 0
        self.image_height = 0 

        self.model = None
        self.framework = ""

    def paths_to_cv2_images(self, paths):
        images = []
        for path in paths:
            img = cv2.imread(path)  
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        return images

    def cv2_images_preprocess(self,cv_images):
        images = []
        for img in cv_images:
            if self.preprocess_settings.denoise_check:
                h = self.preprocess_settings.denoise
                if len(img.shape) == 3:
                    img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
                else:  
                    img = cv2.fastNlMeansDenoising(img, None, h)

            if self.preprocess_settings.blur_check:
                ksize = self.preprocess_settings.blur
                ksize = ksize + 1 if ksize % 2 == 0 else ksize 
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            if self.preprocess_settings.contrast_check:
                alpha = self.preprocess_settings.contrast / 100
                img = cv2.convertScaleAbs(img, alpha=alpha)
            if self.preprocess_settings.brightness_check:
                beta = self.preprocess_settings.brightness
                img = cv2.convertScaleAbs(img, beta=beta)

            images.append(img)
        return images

    def cv2_images_resize(self, cv_images):
        images = []
        for img in cv_images:
            img = cv2.resize(img,(self.image_width,self.image_height))
            images.append(img)
        return images
    
    def backbone_preprocess(self, np_images):
        if self.framework == ".pth" or self.framework == ".pt":
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            preprocess_input = get_preprocessing_fn(
                encoder_name=self.model.backbone,
                pretrained='imagenet'
            )
        else:
            import segmentation_models as sm
            preprocess_input = sm.get_preprocessing(self.model.backbone)

        return preprocess_input(np_images)

    def load_model(self, path):
        if self.framework == ".pth" or self.framework == ".pt":
            from backend.models.pytorch import PyTorchModel
            self.model_uploaded = PyTorchModel.load(path)
        else:
            from backend.models.keras import KerasModel
            self.model_uploaded = KerasModel.load(path)

    def load_weights(self,path):
        from backend.models.pytorch import PyTorchModel

    def evaluate(self):
        if self.predict_method == PredictMethods.SELECTED_MODEL:
            self.model = self.model_selected
        elif self.predict_method == PredictMethods.UPLOADED_MODEL:
            self.model = self.model_uploaded
        elif self.predict_method == PredictMethods.UPLOADED_WEIGHTS:
            self.model = self.weights_uploaded
        if self.model == None:
            popup = PopUpWidget("error", "No model")
            popup.show()
            return
        if len(self.eval_image_paths) == 0:
            popup = PopUpWidget("error", "No Images")
            popup.show()
            return

        self.image_width = self.model.input_size[0]
        self.image_height = self.model.input_size[1]

        self.eval_images = self.paths_to_cv2_images(self.eval_image_paths)
        self.eval_images_preprocessed = self.cv2_images_preprocess(self.eval_images)
        self.eval_images_preprocessed = self.cv2_images_resize(self.eval_images_preprocessed)
        self.eval_images_preprocessed_np = np.array(self.eval_images_preprocessed)
        self.eval_images_preprocessed_np = self.backbone_preprocess(self.eval_images_preprocessed_np)

        if self.framework == ".pth" or self.framework == ".pt":
            import torch
            self.X_val_predict = torch.from_numpy(self.eval_images_preprocessed_np).float()
            print(self.X_val_predict.shape)
            print(self.eval_images_preprocessed_np.shape)
            self.X_val_predict = self.X_val_predict.permute(0, 3, 1, 2).to(self.model.device)
            
            if self.X_val_predict.max() > 1.0:
                self.X_val_predict = self.X_val_predict / 255.0
            
            pad_h = (32 - self.X_val_predict.shape[2] % 32) % 32
            pad_w = (32 - self.X_val_predict.shape[3] % 32) % 32
            if pad_h > 0 or pad_w > 0:
                self.X_val_predict = torch.nn.functional.pad(self.X_val_predict, (0, pad_w, 0, pad_h))
            
            with torch.no_grad():
                predictions = self.model.predict(self.X_val_predict)
                probabilities = torch.sigmoid(predictions)
                                    #(predictions > 0.5)
            binary_predictions = (probabilities).squeeze(1).cpu().numpy()
            
            binary_predictions = (binary_predictions * 255).astype(np.uint8)
        else:
            predictions = self.model.predict(self.X_val_predict)
            binary_predictions = (predictions > 0.5).astype(np.uint8)

        return binary_predictions