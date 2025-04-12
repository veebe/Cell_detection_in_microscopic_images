
import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtCore import QTimer
import numpy as np
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
from PyQt5.QtWidgets import QFileDialog

from backend.backend_types import modelTypes, model_mapping, PYTORCH, KERAS
from backend.setting_classes import ModelSettings, PreprocessingSettings 


class TrainingController:
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

        self.model = None
        self.model_settings = ModelSettings()
        self.preprocess_settings = PreprocessingSettings()

        self.left_timer = QTimer()
        self.right_timer = QTimer()
        self.left_timer.timeout.connect(self.navigate_left)
        self.right_timer.timeout.connect(self.navigate_right)
        self.test_set_visible = True

        self.image_height = 0
        self.image_width = 0

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
    
    def resize_loaded_images(self,image_array):
        images = []
        for img in image_array:
            img = cv2.resize(img,(self.image_width,self.image_height))
            images.append(img)
        return np.array(images)
    
    def resize_loaded_masks(self,mask_array):
        images = []
        for img in mask_array:
            img = cv2.resize(img,(self.image_width,self.image_height))
            images.append(img)
        return np.array(images)[..., np.newaxis]
    
    def apply_prerocessing(self,image_array):
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
        
        
        from sklearn.model_selection import train_test_split
        
        self.ui.training_tab.train_button.setText("Training...") 
        self.ui.training_tab.train_button.setEnabled(False)
        
        self.ui.training_tab.progress_bar.setMaximum(self.model_settings.epochs)

        self.image_height = int(self.ui.training_tab.image_size_dropdown.currentText().split('x')[0]) 
        self.image_width = int(self.ui.training_tab.image_size_dropdown.currentText().split('x')[0]) 

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
                encoder_name= self.model_settings.model_backbone,
                pretrained='imagenet'
            )
        elif self.model_settings.model_framework == KERAS:
            import segmentation_models as sm
            preprocess_input = sm.get_preprocessing(self.model_settings.model_backbone)

        if self.model_settings.model_framework in [PYTORCH, KERAS]:
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
            self.test_loaded_image_array_processed = self.resize_loaded_images(self.apply_prerocessing(self.test_loaded_image_array))
            test_images = self.test_loaded_image_array_processed.astype(np.float32)
            test_images = preprocess_input(test_images) 
            self.X_val_predict = test_images
        else:
            max_index = len(self.val_indices) - 1
            self.ui.training_tab.image_slider.slider.setRange(0, max_index)
            self.current_index = int( max_index / 2 )
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)
        
        if self.model_settings.model_framework == KERAS:
            from backend.models.keras import KerasModel
            self.model = KerasModel(backbone=self.model_settings.model_backbone,input_size=(self.image_height,self.image_width,3))
        elif self.model_settings.model_framework == PYTORCH:
            from backend.models.pytorch import PyTorchModel
            self.model = PyTorchModel(model_type=self.model_settings.model_type,backbone=self.model_settings.model_backbone, input_size=(self.image_height,self.image_width))

        self.model.compile()
        self.model.epochs = self.model_settings.epochs

        if self.model_settings.model_framework == PYTORCH:
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

            train_loader = DataLoader(train_dataset, batch_size=self.model_settings.batch, shuffle=True, num_workers=4, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=self.model_settings.batch, shuffle=False, num_workers=4, pin_memory=True)

            self.model._train(train_loader, val_loader,callbacks = self.callback)

        elif self.model_settings.model_framework == KERAS:
            from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

            datagen = ImageDataGenerator(
                rotation_range=15,
                horizontal_flip=True,
                vertical_flip=True,
                zoom_range=0.1
            )
            train_generator = datagen.flow(
                X_train, 
                y_train, 
                batch_size=self.model_settings.batch
            )

            self.model._train(train_generator, (X_val, y_val),callbacks = self.callback)

    def model_uploaded(self, path):
        import os
        self.predictionController.framework = os.path.splitext(path)[1]
        self.predictionController.load_model(path)
        
    def weights_uploaded(self, path):
        import os
        self.predictionController.framework = os.path.splitext(path)[1]
        self.predictionController.load_weights(path)
        
    def predict(self):
        self.predicted_masks = self.predictionController.evaluate()
        
    def load_eval_images(self, files):
        self.predictionController.eval_image_paths = files

    def on_training_finished(self):
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
                                    #(predictions > 0.5)
            binary_predictions = (probabilities).squeeze(1).cpu().numpy()
            
            binary_predictions = binary_predictions * 255
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
        self.ui.training_tab.train_button.setText("Train and Detect") 
        self.ui.training_tab.train_button.setEnabled(True)
        popup = PopUpWidget("error", error_msg)
        popup.show()

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
        if len(self.image_paths) == 0 or len(self.mask_paths) == 0:# or len(self.processed_images) == 0:
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
        
        self.model_settings.save_settings(values=values)
        self.preprocess_settings.save_settings(values=values)
        print("Model Settings Updated:", self.model_settings.__dict__)
        print("Preprocessing Settings Updated:", self.preprocess_settings.__dict__)

    def download_model(self):
        if not hasattr(self, 'model') or self.model is None:
            popup = PopUpWidget("error", "No model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model_settings.model_framework == KERAS:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.keras", "Keras Files (*.keras);;H5 Files (*.h5);;All Files (*)", options=options)
        elif self.model_settings.model_framework == PYTORCH:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            self.model.save(file_path)

    def download_weights(self):
        if not hasattr(self, 'model') or self.model is None:
            popup = PopUpWidget("error", "No model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model_settings.model_framework == KERAS:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.weights.h5", "H5 Files (*.h5);;All Files (*)", options=options)
        elif self.model_settings.model_framework == PYTORCH:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            self.model.save_weights(file_path)