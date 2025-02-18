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
from frontend.widgets.checkBox import CheckBoxWidget

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()

        self.first_visible_image = None
        self.settings_values = None

        layout = QHBoxLayout(self)

        self.upload_model_button = PurpleButton(text="Upload model")
        self.upload_model_button.clicked.connect(self.upload_model)

        self.decision_label = LabelWidget("OR")

        self.upload_weights_button = PurpleButton(text="Upload weights")
        self.upload_weights_button.clicked.connect(self.upload_weights)

        self.pretrained_models_dropdown_label = LabelWidget("Select Pretrained model")
        self.pretrained_models_dropdown = ComboBoxWidget()
        self.pretrained_models_dropdown.addItems([
            "Mask R-CNN", "StarDist", "DeepCell", "Cellpose", "GAN"
        ])
        self.pretrained_models_dropdown.currentIndexChanged.connect(self.update_model_dropdown)
        self.pretrained_models_dropdown.setCurrentIndex(0)

        self.preprocessing_settings_button = IconButtonWidget("icons/settings-gear-icon.svg")
        self.preprocessing_settings_button.setToolTip("Image Preprocessing")  
        self.preprocessing_settings_button.setFixedSize(20, 20)
        self.preprocessing_settings_button.setIconSize(QSize(20, 20))
        self.preprocessing_settings_button.clicked.connect(self.analysis_settings)

        self.eval_images_drop = DragDropWidget(self, "Drag & Drop images for evaluation here", self.handle_drop)
        self.eval_images_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.predict_button = PurpleButton("Predict")

        self.method_dropdown_label = LabelWidget("Select prediciton method")
        self.method_dropdown = ComboBoxWidget()
        self.method_dropdown.addItems(["Uploaded model","Uploaded weights","Pretrained model"])

        self.method_dropdown.currentIndexChanged.connect(self.update_method_dropdown)
      
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

        predicted_layout = QWidget()
        predicted_layout.setMinimumWidth(600)

        predicted_layout_box = QVBoxLayout(predicted_layout)

        self.threshold_slider = SliderWidget(label_default="threshold",inc_label=False)
        predicted_layout_box.addWidget(self.threshold_slider)
        self.threshold_slider.slider.setRange(0,100)
        self.threshold_slider.slider.setValue(50)

        predicted_image_layout_box = QHBoxLayout()
        self.predicted_image = ImageLabelWidget(label="Mask Preview")
        predicted_image_layout_box.addWidget(self.predicted_image)

        self.segmented_image = ImageLabelWidget(label="Segmentation Preview")
        predicted_image_layout_box.addWidget(self.segmented_image)

        predicted_layout_box.addLayout(predicted_image_layout_box)

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

    def showEvent(self, event):
        super().showEvent(event)
        self.update_model_dropdown()

    def analysis_settings(self):
        from frontend.ui_analysisSettings import AnalysisSettingsDialog
        
        dialog = AnalysisSettingsDialog(self, first_visible_image=self.first_visible_image)
        if self.settings_values is not None:
            dialog.set_all_widget_values(self.settings_values)
        if dialog.exec_() == QDialog.Accepted:
            self.settings_values = dialog.get_all_widget_values()
            if self.settings_values is not None:
                self.controller.predictionController.save_settings(self.settings_values)

    def update_model_dropdown(self):
        self.controller.predictionController.model_selected = self.pretrained_models_dropdown.currentText().lower()

    def update_method_dropdown(self):
        if self.method_dropdown.currentText() == "Uploaded model":
            self.controller.predictionController.predict_method = PredictMethods.UPLOADED_MODEL
        elif self.method_dropdown.currentText() == "Uploaded weights":
            self.controller.predictionController.predict_method = PredictMethods.UPLOADED_WEIGHTS
        elif self.method_dropdown.currentText() == "Pretrained model":
            self.controller.predictionController.predict_method = PredictMethods.SELECTED_MODEL

    def upload_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Model", "", "Model Files (*.h5 *.pt *.pth *.keras)")
        if file_path:
            print(f"Model uploaded: {file_path}")
            self.controller.predictionController.load_model(file_path)

    def upload_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Weights", "", "Weight Files (*weights.h5 *.pt *.pth)")
        if file_path:
            print(f"Weights uploaded: {file_path}")
            self.controller.predictionController.load_weights(file_path)

    def handle_drop(self, files, widget=None):
        if self.controller:
            self.first_visible_image = files[0]
            self.image_slider.slider.setRange(0,len(files)-1)
            self.image_slider.slider.setValue(0)
            if widget == self.eval_images_drop:
                self.controller.predictionController.eval_image_paths = files

    def set_controller(self, controller):
        self.controller = controller
        self.predict_button.clicked.connect(self.controller.predictionController.evaluate)
        self.image_slider.slider.valueChanged.connect(self.controller.predictionController.predict_move_preview)
        self.threshold_slider.slider.valueChanged.connect(self.controller.predictionController.threshold_change)

        self.left_button.pressed.connect(self.controller.predictionController.predict_start_navigate_left)
        self.left_button.released.connect(self.controller.predictionController.predict_stop_navigate)
        self.right_button.pressed.connect(self.controller.predictionController.predict_start_navigate_right)
        self.right_button.released.connect(self.controller.predictionController.predict_stop_navigate)
