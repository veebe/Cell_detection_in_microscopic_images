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
from tensorflow.python.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.python.keras import ImageDataGenerator

class CellDetectionController:
    def __init__(self, ui):
        self.ui = ui
        self.image_paths = []
        self.loaded_image_array = []
        self.mask_paths = []
        self.loaded_mask_array = []
        self.current_index = 0
        self.processed_images = []
        self.val_indices = []
        self.X_val_predict = []

        self.ui.set_controller(self)

    def load_images(self, files):
        if files:
            self.image_paths = files
            self.loaded_image_array = self.convert_loaded_images_to_array(self.image_paths)
            self.ui.image_label.setText("Images uploaded: " + str( len(files) ) )

    def load_masks(self, files):
        if files:
            self.mask_paths = files
            self.loaded_mask_array = self.convert_loaded_masks_to_array(self.mask_paths)
            self.ui.mask_label.setText("Masks uploaded: " + str( len(files) ) ) 

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
        if len(self.loaded_image_array) != len(self.loaded_mask_array):
            popup = PopUpWidget("error", "Mismatched number of images and masks")
            popup.show()
            return
        
        #images = self.loaded_image_array / 255.0
        masks = self.loaded_mask_array / 255.0
        images = self.loaded_image_array.astype(np.float32)  
        #masks = self.loaded_mask_array.astype(np.float32)
        masks = (masks > 0.5).astype(np.float32)

        #X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

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

        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        total_epochs = 10

        self.callback = TrainingProgressCallback(total_epochs)
        self.callback.progress_updated.connect(self.update_process_bar)
        self.callback.training_completed.connect(self.on_training_finished)
        self.callback.training_failed.connect(self.on_training_error)
        self.callback.metrics_updated.connect(self.update_metrics_table)
        
        input_size = (256, 256, 3)
        self.model = sm.Unet('resnet34', input_shape=input_size, encoder_weights='imagenet')

        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        #self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

        #self.thread = TrainingThread(model, X_train, y_train, total_epochs, self.callback)
        #self.thread.start()
        
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

        #self.thread = TrainingThread(self.model, X_train, y_train, total_epochs, self.callback, X_val, y_val)
        self.thread = TrainingThread(
            model = self.model,
            training_data = train_generator, 
            validation_data = (X_val, y_val),
            epochs = total_epochs,
            callback = self.callback
        )
        self.thread.start()

        self.ui.progress_bar.setMaximum(total_epochs)

        self.ui.train_button.setText("Training...") 
        self.ui.train_button.setEnabled(False)
        #history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

        self.X_val_predict = X_val

    def on_training_finished(self):
        predictions = self.model.predict(self.X_val_predict)
        print(f"Predictions shape: {predictions.shape}")
        binary_predictions = (predictions > 0.5).astype(np.uint8)

        self.processed_images = binary_predictions

        self.ui.update_display()

        self.ui.train_button.setEnabled(True)
        self.ui.train_button.setText("Train and detect")
        self.ui.progress_bar.setValue(0)


    def on_training_error(self, error_msg):
        print(f"Error: {error_msg}")

    def update_metrics_table(self, metrics):
        row_position = self.ui.metrics_table.table.rowCount()
        self.ui.metrics_table.table.insertRow(row_position)
        
        self.ui.metrics_table.table.setItem(row_position, 0, QTableWidgetItem(str(metrics['epoch'])))
        self.ui.metrics_table.table.setItem(row_position, 1, QTableWidgetItem(metrics['loss']))
        self.ui.metrics_table.table.setItem(row_position, 2, QTableWidgetItem(metrics['accuracy']))
        self.ui.metrics_table.table.setItem(row_position, 3, QTableWidgetItem(metrics['val_loss']))
        self.ui.metrics_table.table.setItem(row_position, 4, QTableWidgetItem(metrics['val_accuracy']))
        
        self.ui.metrics_table.table.scrollToBottom()

    def update_process_bar(self,value):
        print(value)
        self.ui.progress_bar.setValue(value)

    def display_current_image(self):
        if len(self.image_paths) == 0 or len(self.mask_paths) == 0 or len(self.processed_images) == 0:
            print("Error: No images or masks loaded.")
            return

        if self.current_index >= len(self.image_paths) or self.current_index >= len(self.mask_paths):
            print("Error: Current index out of range.")
            return
        
        original_idx = self.val_indices[self.current_index]

        self.ui.image_drop.display_image(self.image_paths[original_idx])
        self.ui.mask_drop.display_image(self.mask_paths[original_idx])

        predicted_mask = self.processed_images[self.current_index]
        if predicted_mask is not None:
            self.ui.plot.ax.imshow(predicted_mask[..., 0], cmap='gray')
            self.ui.plot.ax.set_title("Predicted Mask")
        else:
            print("Error: No predicted mask available.")
        
        self.ui.plot.canvas.draw()

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