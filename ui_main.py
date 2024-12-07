from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QListWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt


class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Detection Application")
        self.setGeometry(100, 100, 800, 600)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
##origin image block
        self.image_list = QListWidget()
        self.layout.addWidget(self.image_list)

        self.original_image_label = QLabel("Original Image")
        self.original_image_label.setFixedSize(200, 100)
        self.layout.addWidget(self.original_image_label)

        self.original_image_mask = QLabel("Original Image")
        self.original_image_mask.setFixedSize(200, 100)
        self.layout.addWidget(self.original_image_mask)

        self.image_index_label = QLabel("Use left/right arrow keys to navigate images.")
        self.image_index_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_index_label)

        self.load_button = QPushButton("Load Images")
        self.layout.addWidget(self.load_button)

        self.load_masks_button = QPushButton("Load Masks")
        self.layout.addWidget(self.load_masks_button)

        self.detect_button = QPushButton("Train and detect")
        self.layout.addWidget(self.detect_button)
        
        self.figure, self.axes = plt.subplots(1, 3, figsize=(15, 5))  # Three subplots
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        
        """
        self.processed_image_label = QLabel("Processed Image")
        self.layout.addWidget(self.processed_image_label)
        """