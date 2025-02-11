import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from frontend.ui_main import MainUI
from backend.model import IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np
from sklearn.model_selection import train_test_split
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
from backend.training.trainingThread import TrainingThread
import segmentation_models as sm
import segmentation_models_pytorch as smp
import yolov5 as yol
from tensorflow.python.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PyQt5.QtWidgets import QFileDialog

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

    def load_images(self, files):
        if files:
            self.image_paths = files
            self.loaded_image_array = self.convert_loaded_images_to_array(self.image_paths)
            self.ui.training_tab.image_label.setText("Images uploaded: " + str( len(files) ) )

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
            if len(self.test_loaded_mask_array) == 0:
                popup = PopUpWidget("error", "No testing masks loaded")
                if len(self.test_loaded_image_array) == 0:
                    popup = PopUpWidget("error", "No testing images and masks loaded")
                popup.show()
                return
            if len(self.test_loaded_image_array) == 0:
                popup = PopUpWidget("error", "No testing images loaded")
                popup.show()
                return
            if len(self.test_loaded_image_array) != len(self.test_loaded_mask_array):
                popup = PopUpWidget("error", "Mismatched number of testing images and masks")
                popup.show()
                return
        if len(self.loaded_image_array) != len(self.loaded_mask_array):
            popup = PopUpWidget("error", "Mismatched number of images and masks")
            popup.show()
            return
        
        if not self.ui.training_tab.test_set_visible:
            images = self.loaded_image_array.astype(np.float32)  
            masks = (self.loaded_mask_array > 128).astype(np.float32)

            indices = np.arange(len(images))
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
            self.val_indices = val_indices  

            X_train = images[train_indices]
            X_val = images[val_indices]
            y_train = masks[train_indices]
            y_val = masks[val_indices]

            y_train = y_train[..., 0:1]
            y_val = y_val[..., 0:1]

            preprocess_input = sm.get_preprocessing('resnet34')
            X_train = preprocess_input(X_train)
            X_val = preprocess_input(X_val)
        else:
            masks = self.loaded_mask_array / 255.0
            masks = (masks > 0.5).astype(np.float32)
            images = self.loaded_image_array.astype(np.float32)  

            test_images = self.test_loaded_image_array.astype(np.float32)
            # dat moznost pouzit testovaci set ako validacny
            test_masks = self.test_loaded_mask_array / 255.0  
            test_masks = (test_masks > 0.5).astype(np.float32)

            train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

            X_train = train_images
            y_train = train_masks
            X_val = val_images 
            y_val = val_masks

            y_train = y_train[..., 0:1]
            y_val = y_val[..., 0:1]
            
            test_masks = test_masks[..., 0:1]

            preprocess_input = sm.get_preprocessing('resnet34')
            X_train = preprocess_input(X_train)
            X_val = preprocess_input(X_val) 
            test_images = preprocess_input(test_images)
            
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        if not self.ui.training_tab.test_set_visible:
            self.X_val_predict = X_val
        else:
            self.X_val_predict = test_images
        
        total_epochs = 10

        self.callback = TrainingProgressCallback(total_epochs)
        self.callback.progress_updated.connect(self.update_process_bar)
        self.callback.training_completed.connect(self.on_training_finished)
        self.callback.training_failed.connect(self.on_training_error)
        self.callback.metrics_updated.connect(self.update_metrics_table)
        
        input_size = (256, 256, 3)

        
        self.model = sm.Unet('resnet34', input_shape=input_size, encoder_weights='imagenet')
        """
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",   
            encoder_weights="imagenet", 
            in_channels=3,   
            classes=1  
        )
        """

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
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

        self.thread = TrainingThread(
            model = self.model,
            training_data = train_generator, 
            validation_data = (X_val, y_val),
            epochs = total_epochs,
            callback = self.callback
        )
        self.thread.start()

        self.ui.training_tab.progress_bar.setMaximum(total_epochs)

        self.ui.training_tab.train_button.setText("Training...") 
        self.ui.training_tab.train_button.setEnabled(False)

    def on_training_finished(self):
        predictions = self.model.predict(self.X_val_predict)
        print(f"Predictions shape: {predictions.shape}")
        binary_predictions = (predictions > 0.5).astype(np.uint8)

        self.processed_images = predictions

        self.display_current_image()

        self.ui.training_tab.download_button.setEnabled(True)

        self.ui.training_tab.train_button.setEnabled(True)
        self.ui.training_tab.train_button.setText("Train and detect")
        self.ui.training_tab.progress_bar.setValue(0)


    def on_training_error(self, error_msg):
        print(f"Error: {error_msg}")

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
            if len(self.test_image_paths) == 0 or len(self.test_mask_paths) == 0:
                print("Error: No testing images or masks loaded.")
                return

        if self.current_index >= len(self.image_paths) or self.current_index >= len(self.mask_paths):
            print("Error: Current index out of range.")
            return
        
        if not self.ui.training_tab.test_set_visible:
            original_idx = self.val_indices[self.current_index]

            self.ui.training_tab.image_drop.display_image(self.image_paths[original_idx])
            self.ui.training_tab.mask_drop.display_image(self.mask_paths[original_idx])
        else:
            self.ui.training_tab.image_drop.display_image(self.test_image_paths[self.current_index])
            self.ui.training_tab.mask_drop.display_image(self.test_mask_paths[self.current_index])

        predicted_mask = self.processed_images[self.current_index]
        if predicted_mask is not None:
            self.ui.training_tab.plot.ax.imshow(predicted_mask[..., 0], cmap='gray')
            self.ui.training_tab.plot.ax.set_title("Predicted Mask")
        else:
            print("Error: No predicted mask available.")
        
        self.ui.training_tab.plot.canvas.draw()

    def navigate_left(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()

    def navigate_right(self):
        if self.current_index < len(self.processed_images) - 1:
            self.current_index += 1
            self.display_current_image()

    def change_train_test_distribution(self, value):
        print("Slider value changed to:", value)

    def save_model(self):
        if not hasattr(self, 'model') or self.model is None:
            popup = PopUpWidget("error", "No trained model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(None, "Save Model", "model.keras", "Keras Files (*.keras);;H5 Files (*.h5);;All Files (*)", options=options)

        if file_path:
            self.model.save(file_path)
            popup = PopUpWidget("info", f"Model saved to: {file_path}")
            popup.show()
