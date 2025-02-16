from PyQt5.QtWidgets import QLabel, QMessageBox
import os
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent
import cv2
from PyQt5.QtGui import QImage
import numpy as np
from PyQt5.QtCore import Qt


class LabelWidget(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")

class ImageLabelWidget(QLabel):
    def __init__(self, parent=None, label=""):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText(label)
        self.setStyleSheet("""
            border: 2px dashed #aaa;
            color: #ffffff;
            font-size: 10px;            
            font-weight: bold;                    
        """)
        self.original_pixmap = None  
    
    def resizeEvent(self, event):
        self.update_display_size()
        super().resizeEvent(event)

    def update_display_size(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size().width() - 20, self.size().height() - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            self.setText("")
    
    def display_image(self, image_data):
        if isinstance(image_data, np.ndarray):
            if image_data.dtype == np.float32 or image_data.dtype == np.float64:
                image_rgb = (image_data * 255).astype(np.uint8)
            else:
                image_rgb = image_data.astype(np.uint8)

            if len(image_rgb.shape) == 2:
                image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        else:
            QMessageBox.warning(self, "Invalid Data", "The prediction data is not in the correct format.")
            return

        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size().width() - 20, self.size().height() - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_pixmap = pixmap

        self.setPixmap(scaled_pixmap)