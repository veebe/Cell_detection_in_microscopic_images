import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from frontend.ui_main import MainUI
import numpy as np
from sklearn.model_selection import train_test_split
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
import segmentation_models as sm
import yolov5 as yol
from PyQt5.QtWidgets import QFileDialog

from enum import Enum

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256

class modelTypes(Enum):
    UNET = 1
    UNETPP = 2
    DEEPLABV3 = 3
    FNP = 4

class modelFrameworks(Enum):
    KERAS = 1
    PYTORCH = 2

class modelBackbones(Enum):
    RESNET34 = 1
    RESNET50 = 2
    EFFICIENTNET_B3 = 3

model_mapping = {
    "U-Net": modelTypes.UNET,
    "U-Net++": modelTypes.UNETPP,
    "DeepLabV3": modelTypes.DEEPLABV3,
    "FPN": modelTypes.FNP
}

framework_mapping = {
    "Keras": modelFrameworks.KERAS,
    "PyTorch": modelFrameworks.PYTORCH
}

backbone_mapping = {
    "Resnet34": modelBackbones.RESNET34,
    "Resnet50": modelBackbones.RESNET50,
    "EfficientNet-B3": modelBackbones.EFFICIENTNET_B3
}
reverse_backbone_mapping = {v: k for k, v in backbone_mapping.items()}

class ModelSettings:
    def __init__(self):
        self.model_type = modelTypes.UNET
        self.model_framework = modelFrameworks.KERAS
        self.model_backbone = modelBackbones.RESNET34
        self.epochs = 10
        self.val_split = 20

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

class CellDetectionController:
    def __init__(self, ui):
        self.ui = ui
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

        self.ui.set_controller(self)
        self.model = None
        self.model_settings = ModelSettings()
        self.preprocessign_settings = PreprocessingSettings()

        self.left_timer = QTimer()
        self.right_timer = QTimer()
        self.left_timer.timeout.connect(self.navigate_left)
        self.right_timer.timeout.connect(self.navigate_right)
        self.test_set_visible = True

    def load_images(self, files):
        if files:
            self.image_paths = files
            self.loaded_image_array = self.convert_loaded_images_to_array(self.image_paths)
            self.ui.training_tab.image_label.setText("Images uploaded: " + str( len(files) ) )
            if not self.ui.training_tab.test_set_visible:
                self.ui.training_tab.image_slider.slider.setRange(0,len(files)-1)
                self.ui.training_tab.image_slider.slider.setValue(len(files) // 2)

    def load_masks(self, files):
        if files:
            self.mask_paths = files
            self.loaded_mask_array = self.convert_loaded_masks_to_array(self.mask_paths)
            self.ui.training_tab.mask_label.setText("Masks uploaded: " + str( len(files) ) ) 
            print(self.ui.training_tab.test_set_visible)

    def load_test_images(self, files):
        if files:
            self.test_image_paths = files
            self.test_loaded_image_array = self.convert_loaded_images_to_array(self.test_image_paths)
            self.ui.training_tab.test_image_label.setText("Images uploaded: " + str( len(files) ) )
            self.ui.training_tab.image_slider.slider.setRange(0,len(files)-1)
            self.ui.training_tab.image_slider.slider.setValue(len(files) // 2)
    
    def load_test_masks(self, files):
        if files:
            self.test_mask_paths = files
            self.test_loaded_mask_array = self.convert_loaded_masks_to_array(self.test_mask_paths)
            self.ui.training_tab.test_mask_label.setText("Masks uploaded: " + str( len(files) ) )
            

    def convert_loaded_masks_to_array(self, mask_paths):
        masks = []
        for path in mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
                masks.append(mask)
        return np.array(masks)[..., np.newaxis]
    
    def convert_loaded_images_to_array(self, common_image_paths):
        images = []
        for path in common_image_paths:
            img = cv2.imread(path)  
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                images.append(img)
        return np.array(images)

    def train_networks(self):
        if len(self.loaded_mask_array) == 0:
            popup = PopUpWidget("error", "No masks loaded")
            if len(self.loaded_image_array) == 0:
                popup = PopUpWidget("error", "No images and masks loaded")
            popup.show()
            return
        if len(self.loaded_image_array) == 0:
            popup = PopUpWidget("error", "No images loaded")
            popup.show()
            return
        if self.ui.training_tab.test_set_visible:
            if len(self.test_loaded_image_array) == 0:
                popup = PopUpWidget("error", "No testing images and masks loaded")
                popup.show()
                return
        if len(self.loaded_image_array) != len(self.loaded_mask_array):
            popup = PopUpWidget("error", "Mismatched number of images and masks")
            popup.show()
            return
        
        self.ui.training_tab.train_button.setText("Training...") 
        self.ui.training_tab.train_button.setEnabled(False)
        
        self.ui.training_tab.progress_bar.setMaximum(self.model_settings.epochs)
        
        images = self.loaded_image_array.astype(np.float32)  
        masks = (self.loaded_mask_array > 128).astype(np.float32)

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

        preprocess_input = sm.get_preprocessing(reverse_backbone_mapping[self.model_settings.model_backbone].lower())
        X_train = preprocess_input(X_train)
        X_val = preprocess_input(X_val)

        self.X_val_predict = X_val

        print(X_train[0].shape)
        print(y_train[0].shape)

        self.callback = TrainingProgressCallback(self.model_settings.epochs)
        self.callback.progress_updated.connect(self.update_process_bar)
        self.callback.training_completed.connect(self.on_training_finished)
        self.callback.training_failed.connect(self.on_training_error)
        self.callback.metrics_updated.connect(self.update_metrics_table)
        
        if self.ui.training_tab.test_set_visible:
            test_images = self.test_loaded_image_array.astype(np.float32)
            test_images = preprocess_input(test_images)
            self.X_val_predict = test_images
        else:
            max_index = len(self.val_indices) - 1
            self.ui.training_tab.image_slider.slider.setRange(0, max_index)
            self.current_index = int( max_index / 2 )
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)
        
        if self.model_settings.model_framework == modelFrameworks.KERAS:
            from backend.models.keras import KerasModel
            self.model = KerasModel(backbone=reverse_backbone_mapping[self.model_settings.model_backbone])
        elif self.model_settings.model_framework == modelFrameworks.PYTORCH:
            from backend.models.pytorch import PyTorchModel
            self.model = PyTorchModel(backbone=reverse_backbone_mapping[self.model_settings.model_backbone])

        self.model.compile()
        self.model.epochs = self.model_settings.epochs

        if self.model_settings.model_framework == modelFrameworks.PYTORCH:
            from torch.utils.data import TensorDataset, DataLoader
            from backend.training.customDataset import CustomDataset
            import albumentations as A

            train_transform = A.Compose([
                A.Rotate(limit=15, p=0.5),  
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomResizedCrop(
                    height=256,
                    width=256,
                    scale=(0.9, 1.0),
                    ratio=(0.75, 1.33),
                    interpolation=cv2.INTER_LINEAR, 
                    p=1.0 
                )
            ])

            train_dataset = CustomDataset(X_train, y_train, transform=train_transform)
            val_dataset = CustomDataset(X_val, y_val, transform=None) 

            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            self.model._train(train_loader, val_loader,callbacks = self.callback)

        elif self.model_settings.model_framework == modelFrameworks.KERAS:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            datagen = ImageDataGenerator(
                rotation_range=15,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.1
            )
            train_generator = datagen.flow(
                X_train, 
                y_train, 
                batch_size=16
            )

            self.model._train(train_generator, (X_val, y_val),callbacks = self.callback)

    def on_training_finished(self):
        if self.model_settings.model_framework == modelFrameworks.PYTORCH:
            import torch
            print(self.X_val_predict.shape)
            self.X_val_predict = torch.from_numpy(self.X_val_predict).float()
            self.X_val_predict = self.X_val_predict.permute(0, 3, 1, 2).to(self.model.device)
            print(self.X_val_predict.shape)
            
            pad_h = (32 - self.X_val_predict.shape[2] % 32) % 32
            pad_w = (32 - self.X_val_predict.shape[3] % 32) % 32
            self.X_val_predict = torch.nn.functional.pad(self.X_val_predict, (0, pad_w, 0, pad_h))
            print(self.X_val_predict.shape)
            with torch.no_grad():
                predictions = self.model.predict(self.X_val_predict)
            binary_predictions = (
                (predictions > 0.5)
                .squeeze(1)  # Remove channel dimension
                .cpu()
                .numpy()
                .astype(np.uint8)
                    )
        else:
            predictions = self.model.predict(self.X_val_predict)
            binary_predictions = (predictions > 0.5).astype(np.uint8)

        self.processed_images = binary_predictions

        self.display_current_image()

        self.ui.training_tab.download_model_button.setEnabled(True)
        self.ui.training_tab.download_weights_button.setEnabled(True)

        self.ui.training_tab.train_button.setEnabled(True)
        self.ui.training_tab.train_button.setText("Train and detect")
        self.ui.training_tab.progress_bar.setValue(0)


    def on_training_error(self, error_msg):
        print(f"Error: {error_msg}")
        self.ui.training_tab.train_button.setText("Train") 
        self.ui.training_tab.train_button.setEnabled(True)

    def update_metrics_table(self, metrics):
        row_position = self.ui.training_tab.metrics_table.table.rowCount()
        self.ui.training_tab.metrics_table.table.insertRow(row_position)
        
        self.ui.training_tab.metrics_table.table.setItem(row_position, 0, QTableWidgetItem(str(metrics['epoch'])))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 1, QTableWidgetItem(metrics['loss']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 2, QTableWidgetItem(metrics['accuracy']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 3, QTableWidgetItem(metrics['val_loss']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 4, QTableWidgetItem(metrics['val_accuracy']))
        
        self.ui.training_tab.metrics_table.table.scrollToBottom()

    def update_process_bar(self,value):
        print(value)
        self.ui.training_tab.progress_bar.setValue(value)

    def display_current_image(self):
        if len(self.image_paths) == 0 or len(self.mask_paths) == 0 or len(self.processed_images) == 0:
            print("Error: No images or masks loaded.")
            return
        if self.ui.training_tab.test_set_visible:
            if len(self.test_image_paths) == 0:
                print("Error: No testing images loaded.")
                return

        if self.current_index >= len(self.image_paths) or self.current_index >= len(self.mask_paths):
            print("Error: Current index out of range.")
            return
        
        if not self.ui.training_tab.test_set_visible:
            original_idx = self.val_indices[self.current_index]

            self.ui.training_tab.image_drop.display_image(self.image_paths[original_idx])
            self.ui.training_tab.mask_drop.display_image(self.mask_paths[original_idx])
        else:
            self.ui.training_tab.test_image_drop.display_image(self.test_image_paths[self.current_index])
            if len(self.test_mask_paths) != 0:
                self.ui.training_tab.test_mask_drop.display_image(self.test_mask_paths[self.current_index])

        predicted_mask = self.processed_images[self.current_index]
        display_mask = predicted_mask.squeeze()
        if predicted_mask is not None:
            self.ui.training_tab.plot.ax.imshow(display_mask, cmap='gray')
            self.ui.training_tab.plot.ax.set_title("Predicted Mask")
        else:
            print("Error: No predicted mask available.")
        
        self.ui.training_tab.plot.canvas.draw()
    
    def start_navigate_left(self):
        self.navigate_left()  
        self.left_timer.start(100)  

    def start_navigate_right(self):
        self.navigate_right()  
        self.right_timer.start(100) 

    def stop_navigate(self):
        self.left_timer.stop()
        self.right_timer.stop()

    def navigate_left(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)

    def navigate_right(self):
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.display_current_image()
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)
    
    def move_preview(self, value):
        self.current_index = value
        self.display_current_image()
    
    def toggle_training_set(self):
        self.test_set_visible = not self.test_set_visible
        if not self.test_set_visible:
            self.ui.training_tab.image_slider.slider.setRange(0,len(self.image_paths) -1)
            self.ui.training_tab.image_slider.slider.setValue(len(self.image_paths) // 2)
        else:
            self.ui.training_tab.image_slider.slider.setRange(0,len(self.test_image_paths) -1)
            self.ui.training_tab.image_slider.slider.setValue(len(self.test_image_paths) // 2) 

    def save_settings(self, values):

        if 'framework' in values:
            framework_str = values['framework']
            if framework_str in framework_mapping:
                self.model_settings.model_framework = framework_mapping[framework_str]

        if 'model' in values:
            model_str = values['model']
            if model_str in model_mapping:
                self.model_settings.model_type = model_mapping[model_str]

        if 'backbone' in values:
            backbone_str = values['backbone']
            if backbone_str in backbone_mapping:
                self.model_settings.model_backbone = backbone_mapping[backbone_str]

        if 'epochs' in values:
            self.model_settings.epochs = int(values['epochs'])
        
        if 'validation_split' in values:
            self.model_settings.val_split = int(values['validation_split'])

        if 'gaussian_blur' in values:
            gaussian = values['gaussian_blur']
            if isinstance(gaussian, dict) and 'enabled' in gaussian and 'value' in gaussian:
                self.preprocessign_settings.blur_check = gaussian['enabled']
                self.preprocessign_settings.blur = int(gaussian['value'])

        if 'brightness' in values:
            brightness = values['brightness']
            if isinstance(brightness, dict) and 'enabled' in brightness and 'value' in brightness:
                self.preprocessign_settings.brightness_check = brightness['enabled']
                self.preprocessign_settings.brightness = int(brightness['value'])

        if 'contrast' in values:
            contrast = values['contrast']
            if isinstance(contrast, dict) and 'enabled' in contrast and 'value' in contrast:
                self.preprocessign_settings.contrast_check = contrast['enabled']
                self.preprocessign_settings.contrast = int(contrast['value'])

        if 'denoise' in values:
            denoise = values['denoise']
            if isinstance(denoise, dict) and 'enabled' in denoise and 'value' in denoise:
                self.preprocessign_settings.denoise_check = denoise['enabled']
                self.preprocessign_settings.denoise = int(denoise['value'])

        print("Model Settings Updated:", self.model_settings.__dict__)
        print("Preprocessing Settings Updated:", self.preprocessign_settings.__dict__)



    def download_model(self):
        if not hasattr(self, 'model') or self.model is None:
            popup = PopUpWidget("error", "No trained model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model_settings.model_type == modelTypes.UNET:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.keras", "Keras Files (*.keras);;H5 Files (*.h5);;All Files (*)", options=options)

        if self.model_settings.model_type == modelTypes.UNETPP:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            self.model.save(file_path)
            popup = PopUpWidget("info", f"Model saved to: {file_path}")
            popup.show()

    def download_weights(self):
        if not hasattr(self, 'model') or self.model is None:
            popup = PopUpWidget("error", "No trained model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model_settings.model_type == modelTypes.UNET:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.weights.h5", "H5 Files (*.h5);;All Files (*)", options=options)

        if self.model_settings.model_type == modelTypes.UNETPP:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            self.model.save_weights(file_path)
            popup = PopUpWidget("info", f"Weights saved to: {file_path}")
            popup.show()
