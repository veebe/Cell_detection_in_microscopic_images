from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLabel, QComboBox, QSlider, QHBoxLayout,
                             QCheckBox, QSpinBox, QPushButton, QFormLayout)
from PyQt5.QtCore import Qt

class ModelSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setFixedSize(400, 300)
        layout = QVBoxLayout()

        self.model_label = QLabel("Select Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems(["U-Net", "U-Net++", "YOLOAct"])

        self.validation_label = QLabel("Validation Split:")
        self.validation_slider = QSlider(Qt.Horizontal)
        self.validation_slider.setRange(10, 50)  
        self.validation_slider.setValue(20)
        self.validation_slider.setTickInterval(5)
        self.validation_slider.setTickPosition(QSlider.TicksBelow)

        self.preprocessing_label = QLabel("Preprocessing:")
        self.augmentation_checkbox = QCheckBox("Data Augmentation")
        self.normalization_checkbox = QCheckBox("Normalization")

        self.epochs_label = QLabel("Number of Epochs:")
        self.epochs_spinbox = QSpinBox()
        self.epochs_spinbox.setRange(1, 50)
        self.epochs_spinbox.setValue(15)

        self.save_button = QPushButton("Save")
        self.cancel_button = QPushButton("Cancel")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        form_layout = QFormLayout()
        form_layout.addRow(self.model_label, self.model_dropdown)
        form_layout.addRow(self.validation_label, self.validation_slider)
        form_layout.addRow(self.preprocessing_label, self.augmentation_checkbox)
        form_layout.addRow("", self.normalization_checkbox)
        form_layout.addRow(self.epochs_label, self.epochs_spinbox)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(form_layout)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def accept(self):
        print("accepted")

    def rejected(self):
        print("accepted")