import cv2
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from ui_main import MainUI
from model import model, IMAGE_HEIGHT, IMAGE_WIDTH
import numpy as np
from sklearn.model_selection import train_test_split

class CellDetectionController:
    def __init__(self, ui: MainUI):
        self.ui = ui
        self.image_paths = [] 
        self.loaded_image_array = [] 
        self.mask_paths = []
        self.loaded_mask_array = []
        self.current_index = 0  
        self.processed_images = []  

        self.ui.load_button.clicked.connect(self.load_images)
        self.ui.detect_button.clicked.connect(self.train_networks)
        self.ui.load_masks_button.clicked.connect(self.load_masks)

        self.ui.keyPressEvent = self.key_press_event

    def load_masks(self):
        files, _ = QFileDialog.getOpenFileNames(
            self.ui, "Open Image Files", "", "Images (*.png *.jpg *.bmp)"
        )
        if files:
            self.mask_paths = files
            self.loaded_mask_array = self.convert_loaded_images_to_array(self.image_paths)
            self.processed_images = [None] * len(files)  
            #self.display_current_image()     

    def load_images(self):
        # Open file dialog to select multiple images
        files, _ = QFileDialog.getOpenFileNames(
            self.ui, "Open Image Files", "", "Images (*.png *.jpg *.bmp)"
        )
        if files:
            self.ui.image_list.clear()
            self.ui.image_list.addItems(self.image_paths)

            self.image_paths = files
            self.loaded_image_array = self.convert_loaded_images_to_array(self.image_paths)
            self.current_index = 0 
            self.processed_images = [None] * len(files)  
            #self.display_current_image()

    def convert_loaded_images_to_array(self, image_paths):
        images = []
        for path in image_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Resize to uniform dimensions
                images.append(img)
        return np.array(images)           

    def train_networks(self):
        images = self.loaded_image_array / 255.0
        masks = self.loaded_mask_array / 255.0
        masks = (masks > 0.5).astype(np.uint8)

        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

        X_train = X_train[..., np.newaxis]
        X_val = X_val[..., np.newaxis]
        y_train = y_train[..., np.newaxis]
        y_val = y_val[..., np.newaxis]

        history = model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_val, y_val))

        model.evaluate(X_val, y_val)

        predictions = model.predict(X_val)

        binary_predictions = (predictions > 0.5).astype(np.uint8)

        self.processed_images = [binary_predictions[i, ..., 0] for i in range(len(binary_predictions))]
        self.display_current_image()

    def display_current_image(self):
        if not self.image_paths:
            return

        image_path = self.image_paths[self.current_index]
        original_image = cv2.imread(image_path)
        #original_pixmap = self.convert_to_pixmap(original_image)
        #self.ui.original_image_label.setPixmap(original_pixmap)

        true_mask = cv2.imread(self.mask_paths[self.current_index])
        predicted_mask = self.processed_images[self.current_index]
        
        processed_image = self.processed_images[self.current_index]
        if processed_image is not None:
            """
            self.ui.ax.clear()
            self.ui.ax.imshow(processed_image)
            self.ui.ax.set_title(f"Detected Cells in image \n{image_path}")
            self.ui.canvas.draw()
            """

            self.ui.axes[0].clear()
            self.ui.axes[0].imshow(original_image, cmap='gray')
            self.ui.axes[0].set_title("Input Image")

            self.ui.axes[1].clear()
            self.ui.axes[1].imshow(true_mask, cmap='gray')
            self.ui.axes[1].set_title("True Mask")

            self.ui.axes[2].clear()
            self.ui.axes[2].imshow(predicted_mask, cmap='gray')
            self.ui.axes[2].set_title("Predicted Mask")

            self.ui.canvas.draw()
        
        self.ui.image_index_label.setText(
            f"Image {self.current_index + 1} of {len(self.image_paths)}"
        )

    def key_press_event(self, event):
        # Navigate through images using left/right arrow keys
        if event.key() == Qt.Key_Right:
            self.current_index = (self.current_index + 1) % len(self.image_paths)
            self.display_current_image()
        elif event.key() == Qt.Key_Left:
            self.current_index = (self.current_index - 1) % len(self.image_paths)
            self.display_current_image()

    @staticmethod
    def convert_to_pixmap(image):
        height, width, channels = image.shape
        bytes_per_line = channels * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(q_image)
