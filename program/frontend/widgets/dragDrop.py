import os
from PyQt5.QtGui import QPixmap, QDragEnterEvent, QDropEvent
import cv2
from PyQt5.QtWidgets import QLabel, QMessageBox
from PyQt5.QtGui import QImage
from PyQt5.QtCore import Qt

class DragDropWidget(QLabel):
    def __init__(self, parent=None, label="", drop_handler=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText(label)
        self.setStyleSheet("""
            border: 2px dashed #aaa;
            color: #ffffff;
            font-size: 10px;            
            font-weight: bold;                    
        """)
        self.drop_handler = drop_handler 
        self.original_pixmap = None  

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            if os.path.isdir(files[0]):
                folder_path = files[0]
                image_files = self.get_image_files_from_folder(folder_path)
                if image_files:
                    self.display_image(image_files[0]) 
                    self.drop_handler(image_files, self)
                else:
                    QMessageBox.warning(self, "No Images", "The folder does not contain any supported image files.")
            else:
                self.display_image(files[0])  
                self.drop_handler(files, self) 
        event.acceptProposedAction()
    
    def resizeEvent(self, event):
        self.update_display_size()
        super().resizeEvent(event)

    def update_display_size(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(self.size().width() - 20, self.size().height() - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)
            self.setText("")
    
    def display_image(self, file_path):
        image = cv2.imread(file_path)
        if image is None:
            QMessageBox.warning(self, "Invalid Image", "The file is not a valid image.")
            return
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.size().width() - 20, self.size().height() - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_pixmap = pixmap

        self.setPixmap(scaled_pixmap)
        self.setText("")

    def get_image_files_from_folder(self, folder_path):
        supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')
        image_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(supported_formats):
                    image_files.append(os.path.join(root, file))
        return image_files