import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from backend.setting_classes import ModelSettings, PreprocessingSettings
from backend.backend_types import PYTORCH, KERAS

class TrainingModel:
    def __init__(self):
        self.image_paths = []
        self.loaded_image_array = []
        self.mask_paths = []
        self.loaded_mask_array = []
        
        self.test_image_paths = []
        self.test_loaded_image_array = []
        self.test_mask_paths = []
        self.test_loaded_mask_array = []
        self.using_dedicated_testing_set = True

        self.current_index = 0
        self.processed_images = []
        self.val_indices = []
        self.X_val_predict = []

        self.model = None
        self.model_settings = ModelSettings()
        self.preprocess_settings = PreprocessingSettings()
        
        self.image_height = 0
        self.image_width = 0

    def load_images(self, files):
        if files:
            self.image_paths = files
            self.loaded_image_array = self.convert_loaded_images_to_array(self.image_paths)
            return len(files)
        return 0

    def load_masks(self, files):
        if files:
            self.mask_paths = files
            self.loaded_mask_array = self.convert_loaded_masks_to_array(self.mask_paths)
            return len(files)
        return 0

    def load_test_images(self, files):
        if files:
            self.test_image_paths = files
            self.test_loaded_image_array = self.convert_loaded_images_to_array(self.test_image_paths)
            return len(files)
        return 0
    
    def load_test_masks(self, files):
        if files:
            self.test_mask_paths = files
            self.test_loaded_mask_array = self.convert_loaded_masks_to_array(self.test_mask_paths)
            return len(files)
        return 0

    def convert_loaded_masks_to_array(self, mask_paths):
        masks = []
        for path in mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                masks.append(mask)
        return masks
    
    def convert_loaded_images_to_array(self, common_image_paths):
        images = []
        for path in common_image_paths:
            img = cv2.imread(path)  
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        return images
    
    def resize_loaded_images(self, image_array):
        images = []
        for img in image_array:
            img = cv2.resize(img, (self.image_width, self.image_height))
            images.append(img)
        return np.array(images)
    
    def resize_loaded_masks(self, mask_array):
        images = []
        for img in mask_array:
            img = cv2.resize(img, (self.image_width, self.image_height))
            images.append(img)
        return np.array(images)[..., np.newaxis]
    
    def apply_prerocessing(self, image_array):
        images = []
        for img in image_array:
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

            if self.preprocess_settings.denoise_check:
                h = self.preprocess_settings.denoise
                if len(img.shape) == 3:
                    img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
                else:  
                    img = cv2.fastNlMeansDenoising(img, None, h)

            images.append(img)
        return images
    
    def validate_train_data(self):
        if len(self.loaded_mask_array) == 0:
            return False, "No masks loaded"
        if len(self.loaded_image_array) == 0:
            return False, "No images loaded"
        if self.using_dedicated_testing_set:
            if len(self.test_loaded_image_array) == 0:
                return False, "No testing images loaded"
        if len(self.loaded_image_array) != len(self.loaded_mask_array):
            return False, "Mismatched number of images and masks"
        return True, ""
    
    def prepare_training_data(self, image_size):
        self.image_height = image_size[0]
        self.image_width = image_size[1]

        self.loaded_image_array_processed = self.resize_loaded_images(self.apply_prerocessing(self.loaded_image_array))
        self.loaded_mask_array_processed = self.resize_loaded_masks(self.loaded_mask_array)

        images = self.loaded_image_array_processed.astype(np.float32)  
        masks = (self.loaded_mask_array_processed > 128).astype(np.float32)

        indices = np.arange(len(images))
        val_split = self.model_settings.val_split/100
        train_indices, val_indices = train_test_split(indices, test_size=val_split, random_state=42)
        self.val_indices = val_indices  

        X_train = images[train_indices]
        X_val = images[val_indices]
        y_train = masks[train_indices]
        y_val = masks[val_indices]

        y_train = y_train[..., 0:1]
        y_val = y_val[..., 0:1]

        if self.model_settings.model_framework == PYTORCH:
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            preprocess_input = get_preprocessing_fn(
                encoder_name=self.model_settings.model_backbone,
                pretrained='imagenet'
            )
        elif self.model_settings.model_framework == KERAS:
            import os
            os.environ["SM_FRAMEWORK"] = "tf.keras"
            import segmentation_models as sm
            preprocess_input = sm.get_preprocessing(self.model_settings.model_backbone)

        if self.model_settings.model_framework in [PYTORCH, KERAS]:
            X_train = preprocess_input(X_train)
            X_val = preprocess_input(X_val)

        if self.using_dedicated_testing_set:
            self.test_loaded_image_array_processed = self.resize_loaded_images(self.apply_prerocessing(self.test_loaded_image_array))
            test_images = self.test_loaded_image_array_processed.astype(np.float32)
            test_images = preprocess_input(test_images) 
            self.X_val_predict = test_images
        else:
            self.X_val_predict = X_val
        
        return X_train, X_val, y_train, y_val
    
    def create_model(self):
        if self.model_settings.model_framework == KERAS:
            from backend.models.keras import KerasModel
            self.model = KerasModel(backbone=self.model_settings.model_backbone,
                                   input_size=(self.image_height, self.image_width, 3))
        elif self.model_settings.model_framework == PYTORCH:
            from backend.models.pytorch import PyTorchModel
            self.model = PyTorchModel(model_type=self.model_settings.model_type,
                                     backbone=self.model_settings.model_backbone, 
                                     input_size=(self.image_height, self.image_width))

        self.model.compile()
        self.model.epochs = self.model_settings.epochs
        return self.model
    
    def setup_pytorch_training(self, X_train, y_train, X_val, y_val):
        import torch
        from torch.utils.data import DataLoader
        from backend.training.customDataset import CustomDataset
        import albumentations as A

        train_transform = A.Compose([
            A.Rotate(limit=15, p=0.5),  
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomResizedCrop(
                height=self.image_height,
                width=self.image_width,
                scale=(0.9, 1.0),
                ratio=(0.75, 1.33),
                interpolation=cv2.INTER_LINEAR, 
                p=1.0 
            )
        ])

        train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
        val_dataset = CustomDataset(X_val, y_val, transform=None) 

        train_loader = DataLoader(train_dataset, batch_size=self.model_settings.batch, 
                                 shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.model_settings.batch, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, val_loader
    
    def setup_keras_training(self, X_train, y_train):
        from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
        """
        datagen = ImageDataGenerator(
            rotation_range=15,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.1
        )
        """
        datagen = ImageDataGenerator(
            rotation_range=30,         
            width_shift_range=0.15,    
            height_shift_range=0.15,   
            shear_range=0.1,           
            zoom_range=0.2,            
            horizontal_flip=True,      
            vertical_flip=True,
            fill_mode='nearest',       
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.1,   
        )
        
        train_generator = datagen.flow(
            X_train, 
            y_train, 
            batch_size=self.model_settings.batch
        )
        
        return train_generator
    
    def predict_on_validation(self):
        if self.model_settings.model_framework == PYTORCH:
            import torch
            self.X_val_predict = torch.from_numpy(self.X_val_predict).float()
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
            binary_predictions = (probabilities).squeeze(1).cpu().numpy()
            
            binary_predictions = binary_predictions * 255
        else:
            predictions = self.model.predict(self.X_val_predict)
            binary_predictions = (predictions > 0.5).astype(np.uint8)

        self.processed_images = binary_predictions
        return binary_predictions
    
    def get_validation_image(self, index):
        if 0 <= index < len(self.processed_images):
            return self.processed_images[index]
        return None
    
    def get_original_image_and_mask(self, index):
        if self.using_dedicated_testing_set:
            if 0 <= index < len(self.test_image_paths):
                test_image = self.test_image_paths[index] if index < len(self.test_image_paths) else None
                test_mask = self.test_mask_paths[index] if index < len(self.test_mask_paths) else None
                return test_image, test_mask
            return None, None
        else:
            if 0 <= index < len(self.val_indices):
                original_idx = self.val_indices[index]
                if original_idx < len(self.image_paths) and original_idx < len(self.mask_paths):
                    return self.image_paths[original_idx], self.mask_paths[original_idx]
            return None, None
    
    def save_model(self, path):
        if self.model:
            self.model.save(path)
            return True
        return False
    
    def save_weights(self, path):
        if self.model:
            self.model.save_weights(path)
            return True
        return False