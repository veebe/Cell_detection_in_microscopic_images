from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy, QToolTip, QPushButton, QSplitter, QDialog, QFormLayout, QFileDialog
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor
from frontend.widgets.button import PurpleButton
from frontend.widgets.dragDrop import DragDropWidget
from frontend.widgets.plot import PlotWidget
from frontend.widgets.table import TableWidget
from frontend.widgets.processBar import ProgressBarWidget
from frontend.widgets.icon import IconButtonWidget
from frontend.widgets.splitter import SplitterWidget
from frontend.widgets.slider import SliderWidget
from frontend.widgets.combobox import ComboBoxWidget
from frontend.widgets.label import LabelWidget, ImageLabelWidget
from backend.backend_predict import PredictMethods

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)

        self.upload_model_button = PurpleButton(text="Upload model")
        self.upload_model_button.clicked.connect(self.upload_model)

        self.decision_label = LabelWidget("OR")

        self.upload_weights_button = PurpleButton(text="Upload weights")
        self.upload_weights_button.clicked.connect(self.upload_weights)

        self.pretrained_models_dropdown_label = LabelWidget("Select Pretrained model")
        self.pretrained_models_dropdown = ComboBoxWidget()
        self.pretrained_models_dropdown.addItems([
            "ResNet-50", "ResNet-101", "VGG16", "VGG19", "MobileNetV2", "EfficientNet-B0", "EfficientNet-B7"
        ])
        self.pretrained_models_dropdown.currentIndexChanged.connect(self.update_model_dropdown)

        self.preprocessing_settings_button = IconButtonWidget("icons/settings-gear-icon.svg")
        self.preprocessing_settings_button.setToolTip("Image Preprocessing")  
        self.preprocessing_settings_button.setFixedSize(20, 20)
        self.preprocessing_settings_button.setIconSize(QSize(20, 20))

        self.eval_images_drop = DragDropWidget(self, "Drag & Drop images for evaluation here", self.handle_drop)
        self.eval_images_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.predict_button = PurpleButton("Predict")

        self.method_dropdown_label = LabelWidget("Select prediciton method")
        self.method_dropdown = ComboBoxWidget()
        self.method_dropdown.addItems(["Uploaded model","Uploaded weights","Pretrained model"])
      
        form_layout_widget = QWidget()
        form_layout = QFormLayout(form_layout_widget)
        form_layout.addRow(self.upload_model_button)
        form_layout.addRow(self.decision_label)
        form_layout.addRow(self.upload_weights_button)
        form_layout.addRow(self.pretrained_models_dropdown_label, self.pretrained_models_dropdown)
        form_layout.addRow(self.method_dropdown_label,self.method_dropdown)
        form_layout.addRow(self.preprocessing_settings_button)
        form_layout.addRow(self.eval_images_drop)
        form_layout.addRow(self.predict_button)

        splitter = SplitterWidget(Qt.Horizontal)
        splitter.addWidget(form_layout_widget)
        """
        self.predicted_image = QLabel("Image Preview")
        self.predicted_image.setAlignment(Qt.AlignCenter)
        self.predicted_image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.predicted_image.setStyleSheet(""
            border: 2px dashed #aaa;
            color: #ffffff;
            font-size: 10px;            
            font-weight: bold;                    
        "")
        """
        predicted_layout = QWidget()
        predicted_layout.setMinimumWidth(400)
        predicted_layout_box = QVBoxLayout(predicted_layout)
        self.predicted_image = ImageLabelWidget("Image Preview")
        predicted_layout_box.addWidget(self.predicted_image)
        
        self.image_slider = SliderWidget()
        predicted_layout_box.addWidget(self.image_slider)
 
        self.navigation_layout = QHBoxLayout()
        self.left_button = PurpleButton("<")
        self.navigation_layout.addWidget(self.left_button)
        self.right_button = PurpleButton(">")
        self.navigation_layout.addWidget(self.right_button)
        predicted_layout_box.addLayout(self.navigation_layout)

        splitter.addWidget(predicted_layout)

        layout.addWidget(splitter)

    def update_model_dropdown(self):
        if self.method_dropdown.currentText() == "Uploaded model":
            self.controller.predictionController.self.predict_method = PredictMethods.UPLOADED_MODEL
        elif self.method_dropdown.currentText() == "Uploaded weights":
            self.controller.predictionController.self.predict_method = PredictMethods.UPLOADED_WEIGHTS
        elif self.method_dropdown.currentText() == "Pretrained model":
            self.controller.predictionController.self.predict_method = PredictMethods.SELECTED_MODEL

    def upload_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Model", "", "Model Files (*.h5 *.pt *.pth)")
        if file_path:
            print(f"Model uploaded: {file_path}")
            self.controller.model_uploaded(file_path)

    def upload_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Weights", "", "Weight Files (*weights.h5 *.pt *.pth)")
        if file_path:
            print(f"Weights uploaded: {file_path}")
            self.controller.weights_uploaded(file_path)

    def handle_drop(self, files, widget=None):
        if self.controller:
            self.image_slider.slider.setRange(0,len(files)-1)
            if widget == self.eval_images_drop:
                self.controller.load_eval_images(files)

    def set_controller(self, controller):
        self.controller = controller
        self.predict_button.clicked.connect(self.controller.predict)