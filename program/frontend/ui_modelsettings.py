from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QHBoxLayout, QWidget, QFormLayout, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QEvent
from frontend.widgets.label import LabelWidget
from frontend.widgets.slider import SliderWidget
from frontend.widgets.combobox import ComboBoxWidget
from frontend.widgets.button import PurpleButton
from frontend.widgets.tabs import TabWidget
from frontend.widgets.checkBox import CheckBoxWidget
from frontend.widgets.splitter import SplitterWidget

import numpy as np
import cv2



class ModelSettingsDialog(QDialog):
    def __init__(self, parent=None, first_visible_image=None):
        super().__init__(parent)
        self.setWindowTitle("Model Settings")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinimizeButtonHint)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self.current_image = first_visible_image
        self.modified_image = self.load_image()

        self.tabs = TabWidget() 

        self.setup_model_settings_ui()
        self.setup_preprocessing_ui()

        self.tabs.addTab(self.model_settings_widget, "Model Settings")
        self.tabs.addTab(self.preprocess_widget, "Preprocessing")

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        button_layout = QHBoxLayout()
        self.save_button = PurpleButton("Save")
        self.cancel_button = PurpleButton("Cancel")
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

        self.framework_dropdown.currentIndexChanged.connect(self.update_model_dropdown)
        self.update_model_dropdown()

    def showEvent(self, event):
        super().showEvent(event)
        self.display_image(self.load_image())

    def setup_model_settings_ui(self):
        self.model_settings_widget = QWidget()
        layout = QVBoxLayout()

        self.framework_label = LabelWidget("Select Framework:")
        self.framework_dropdown = ComboBoxWidget()
        self.framework_dropdown.addItems(["Keras", "PyTorch"])

        self.model_label = LabelWidget("Select Model:")
        self.model_dropdown = ComboBoxWidget()

        self.backbone_label = LabelWidget("Select Backbone:")
        self.backbone_dropdown = ComboBoxWidget()
        
        self.validation_label = LabelWidget("Validation Split:")
        self.validation_slider = SliderWidget(Qt.Horizontal, inc_label=False, label_default="Validation set")
        self.validation_slider.slider.setRange(5, 50)
        self.validation_slider.slider.setValue(20)
        self.validation_slider.slider.setTickInterval(1)

        self.epochs_label = LabelWidget("Number of Epochs:")
        self.epochs_slider = SliderWidget(Qt.Horizontal, inc_label=False, label_default="Number of epochs", percent=False)
        self.epochs_slider.slider.setRange(1, 50)
        self.epochs_slider.slider.setValue(15)
        self.epochs_slider.slider.setTickInterval(1)

        self.batch_label = LabelWidget("Batch size:")
        self.batch_slider = SliderWidget(Qt.Horizontal, inc_label=False, label_default="Batch size", percent=False, power_of_two=True)
        self.batch_slider.slider.setRange(1, 6)
        self.batch_slider.set_power_value(2)
        
        self.batch_slider.slider.setSliderPosition(4)  

        self.reset_model_button = PurpleButton("Reset")
        self.reset_model_button.clicked.connect(self.reset_model)

        form_layout = QFormLayout()
        form_layout.addRow(self.framework_label, self.framework_dropdown)
        form_layout.addRow(self.model_label, self.model_dropdown)
        form_layout.addRow(self.backbone_label, self.backbone_dropdown)
        form_layout.addRow(self.validation_label, self.validation_slider)
        form_layout.addRow(self.epochs_label, self.epochs_slider)
        form_layout.addRow(self.batch_label, self.batch_slider)
        form_layout.addRow(self.reset_model_button)

        layout.addLayout(form_layout)
        self.model_settings_widget.setLayout(layout)

    def reset_model(self):
        self.framework_dropdown.setCurrentIndex(0)
        self.update_model_dropdown()
        self.backbone_dropdown.setCurrentIndex(0)
        self.validation_slider.slider.setValue(20)
        self.epochs_slider.slider.setValue(15)
        self.batch_slider.slider.setSliderPosition(4)

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

        self.preview_label = QLabel("Image Preview")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("""
            border: 2px dashed #aaa;
            color: #ffffff;
            font-size: 10px;            
            font-weight: bold;                    
        """)

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

    def update_model_dropdown(self):
        framework = self.framework_dropdown.currentText()
        self.model_dropdown.clear()
        self.backbone_dropdown.clear()

        if framework == "Keras":
            self.model_dropdown.addItems(["U-Net"])
            self.backbone_dropdown.addItems(['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 'seresnext50', 'seresnext101',
                                             'senet154', 'resnext50', 'resnext101', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet201',
                                             'inceptionresnetv2', 'inceptionv3', 'mobilenet', 'mobilenetv2', 'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5'])

        elif framework == "PyTorch":
            self.model_dropdown.addItems(["U-Net", "U-Net++", "DeepLabV3", "FPN"])
            self.backbone_dropdown.addItems([
                                                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                'resnext50_32x4d', 'resnext101_32x8d',
                                                'densenet121', 'densenet169', 'densenet161', 'densenet201',
                                                'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
                                                'mobilenet_v2',
                                                'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3',
                                                'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7',
                                                'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
                                                'inceptionresnetv2', 'inceptionv4',
                                                'xception',
                                                'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d', 'timm-resnest101e',
                                                'timm-regnetx_002', 'timm-regnetx_032', 'timm-regnety_002', 'timm-regnety_032',
                                                'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d'
                                            ])

    def accept(self):
        print("Accepted")
        super().accept()

    def reject(self):
        print("Rejected")
        super().reject()

    def load_image(self):
        if not self.current_image or not isinstance(self.current_image, str):
            return

        img = cv2.imread(self.current_image)
        if img is None:
            print("Error: Could not load image.")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def display_image(self, img=None):
        if img is None:
            img = self.modified_image
            if img is None:
                return
        height, width, channel = img.shape
        bytes_per_line = 3 * width

        qt_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.preview_label.size().width() - 20, self.preview_label.size().height() - 20, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap)

    def eventFilter(self, obj, event):
        if obj == self.preview_label and event.type() == QEvent.Resize:
            self.display_image()
            return True  

        return super().eventFilter(obj, event)

    def update_preview(self):
        img = self.load_image()
        if img is None:
            return

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

        if self.denoise_check.isChecked():
            h = self.denoise_slider.slider.value()
            if len(img.shape) == 3:
                img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
            else:  
                img = cv2.fastNlMeansDenoising(img, None, h)

        self.modified_image = img
        self.display_image(self.modified_image)


    def get_all_widget_values(self):
        values = {}

        values['framework'] = self.framework_dropdown.currentText()
        values['model'] = self.model_dropdown.currentText()
        values['backbone'] = self.backbone_dropdown.currentText()
        values['validation_split'] = self.validation_slider.slider.value()
        values['epochs'] = self.epochs_slider.slider.value()
        values['batch'] = self.batch_slider.slider.value()

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
        if 'framework' in values:
            framework = values['framework']
            if framework in [self.framework_dropdown.itemText(i) for i in range(self.framework_dropdown.count())]:
                self.framework_dropdown.setCurrentText(framework)
        if 'model' in values:
            model = values['model']
            self.model_dropdown.setCurrentText(model)
        if 'backbone' in values:
            backbone = values['backbone']
            self.backbone_dropdown.setCurrentText(backbone)
        if 'validation_split' in values:
            self.validation_slider.slider.setValue(values['validation_split'])
        if 'epochs' in values:
            self.epochs_slider.slider.setValue(values['epochs'])
        if 'batch' in values:
            self.batch_slider.slider.setValue(values['batch'])

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