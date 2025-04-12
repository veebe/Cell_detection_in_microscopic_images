from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt
from frontend.widgets.slider import SliderWidget
from frontend.widgets.button import PurpleButton
from frontend.widgets.checkBox import CheckBoxWidget
from frontend.widgets.splitter import SplitterWidget
from frontend.widgets.label import ImageLabelWidget

import numpy as np
import cv2


class AnalysisSettingsDialog(QDialog):
    def __init__(self, parent=None, first_visible_image=None):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.first_visible_image = first_visible_image
        self.current_image = self.first_visible_image

        self.setup_preprocessing_ui() 

        layout = QVBoxLayout()   
        layout.addWidget(self.preprocess_widget)

        button_layout = QHBoxLayout()
        self.save_button = PurpleButton("Save")
        self.cancel_button = PurpleButton("Cancel")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def showEvent(self, event):
        super().showEvent(event)
        self.preview_label.display_image(self.update_preview())

    def accept(self):
        print("Accepted")
        super().accept()

    def reject(self):
        print("Rejected")
        super().reject()

    def reset_preprocess(self):
        self.gaussian_check.setChecked(False)
        self.gaussian_slider.slider.setValue(1)
        self.brightness_slider.slider.setValue(0)
        self.contrast_slider.slider.setValue(100)
        self.denoise_check.setChecked(False)
        self.denoise_slider.slider.setValue(1)
        self.brightness_check.setChecked(False)
        self.contrast_check.setChecked(False)
        self.update_preview()

    def setup_preprocessing_ui(self):
        self.preprocess_widget = QWidget()
        main_layout = QHBoxLayout()  

        preprocessing_layout_container = QWidget() 
        preprocessing_layout_container.setMinimumWidth(250) 
        preprocessing_layout = QVBoxLayout(preprocessing_layout_container)

        self.gaussian_check = CheckBoxWidget("Gaussian Blur")
        self.gaussian_slider = SliderWidget(Qt.Horizontal,inc_label=False,label_default="Gaussian blur")
        self.gaussian_slider.slider.setRange(1, 100)
        self.gaussian_slider.slider.setSingleStep(2)
        self.gaussian_slider.slider.setValue(1)
        preprocessing_layout.addWidget(self.gaussian_check)
        preprocessing_layout.addWidget(self.gaussian_slider)

        self.brightness_check = CheckBoxWidget("Brightness")
        self.brightness_slider = SliderWidget(Qt.Horizontal,inc_label=False,label_default="Brightness")
        self.brightness_slider.slider.setRange(-100, 100)
        self.brightness_slider.slider.setValue(0)
        preprocessing_layout.addWidget(self.brightness_check)
        preprocessing_layout.addWidget(self.brightness_slider)

        self.contrast_check = CheckBoxWidget("Contrast")
        self.contrast_slider = SliderWidget(Qt.Horizontal,inc_label=False,label_default="Contrast",percent=False)
        self.contrast_slider.slider.setRange(0, 300)
        self.contrast_slider.slider.setValue(100)
        preprocessing_layout.addWidget(self.contrast_check)
        preprocessing_layout.addWidget(self.contrast_slider)

        self.denoise_check = CheckBoxWidget("Denoise")
        self.denoise_slider = SliderWidget(Qt.Horizontal,inc_label=False,label_default="Denoise")
        self.denoise_slider.slider.setRange(1, 100)
        self.denoise_slider.slider.setValue(1)
        preprocessing_layout.addWidget(self.denoise_check)
        preprocessing_layout.addWidget(self.denoise_slider)

        self.reset_pre_button = PurpleButton("Reset")
        self.reset_pre_button.clicked.connect(self.reset_preprocess)
        preprocessing_layout.addWidget(self.reset_pre_button)

        self.preview_label = ImageLabelWidget(label="Image Preview")
        self.preview_label.setMinimumWidth(300)

        self.preview_label.installEventFilter(self)

        splitter = SplitterWidget(Qt.Horizontal)
        splitter.addWidget(preprocessing_layout_container)
        splitter.addWidget(self.preview_label)

        main_layout.addWidget(splitter)
        self.preprocess_widget.setLayout(main_layout)

        self.gaussian_check.stateChanged.connect(self.update_preview)
        self.gaussian_slider.slider.valueChanged.connect(self.update_preview)
        self.brightness_slider.slider.valueChanged.connect(self.update_preview)
        self.contrast_slider.slider.valueChanged.connect(self.update_preview)
        self.denoise_check.stateChanged.connect(self.update_preview)
        self.denoise_slider.slider.valueChanged.connect(self.update_preview)
        self.brightness_check.stateChanged.connect(self.update_preview)
        self.contrast_check.stateChanged.connect(self.update_preview)

    def load_image(self):
        if not self.current_image or not isinstance(self.current_image, str):
            return

        img = cv2.imread(self.current_image)
        if img is None:
            print("Error: Could not load image.")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def update_preview(self):
        img = self.load_image()
        if img is None:
            return
        
        if self.denoise_check.isChecked():
            h = self.denoise_slider.slider.value()
            if len(img.shape) == 3:
                img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
            else:  
                img = cv2.fastNlMeansDenoising(img, None, h)

        if self.gaussian_check.isChecked():
            ksize = self.gaussian_slider.slider.value()
            ksize = ksize + 1 if ksize % 2 == 0 else ksize 
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        if self.contrast_check.isChecked():
            alpha = self.contrast_slider.slider.value() / 100
            img = cv2.convertScaleAbs(img, alpha=alpha)
        if self.brightness_check.isChecked():
            beta = self.brightness_slider.slider.value()
            img = cv2.convertScaleAbs(img, beta=beta)

        self.modified_image = img
        self.preview_label.display_image(self.modified_image)

    def get_all_widget_values(self):
        values = {}
        values['gaussian_blur'] = {
            'enabled': self.gaussian_check.isChecked(),
            'value': self.gaussian_slider.slider.value()
        }
        values['brightness'] = {
            'enabled': self.brightness_check.isChecked(),
            'value': self.brightness_slider.slider.value()
        }
        values['contrast'] = {
            'enabled': self.contrast_check.isChecked(),
            'value': self.contrast_slider.slider.value()
        }
        values['denoise'] = {
            'enabled': self.denoise_check.isChecked(),
            'value': self.denoise_slider.slider.value()
        }

        return values
    
    def set_all_widget_values(self, values):
        if 'gaussian_blur' in values:
            gaussian = values['gaussian_blur']
            if isinstance(gaussian, dict) and 'enabled' in gaussian and 'value' in gaussian:
                self.gaussian_check.setChecked(gaussian['enabled'])
                self.gaussian_slider.slider.setValue(gaussian['value'])

        if 'brightness' in values:
            brightness = values['brightness']
            if isinstance(brightness, dict) and 'enabled' in brightness and 'value' in brightness:
                self.brightness_check.setChecked(brightness['enabled'])
                self.brightness_slider.slider.setValue(brightness['value'])

        if 'contrast' in values:
            contrast = values['contrast']
            if isinstance(contrast, dict) and 'enabled' in contrast and 'value' in contrast:
                self.contrast_check.setChecked(contrast['enabled'])
                self.contrast_slider.slider.setValue(contrast['value'])

        if 'denoise' in values:
            denoise = values['denoise']
            if isinstance(denoise, dict) and 'enabled' in denoise and 'value' in denoise:
                self.denoise_check.setChecked(denoise['enabled'])
                self.denoise_slider.slider.setValue(denoise['value'])