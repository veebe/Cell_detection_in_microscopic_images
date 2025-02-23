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

        self.upload_model_widget = QWidget()
        self.upload_model_layout = QHBoxLayout(self.upload_model_widget)
        self.upload_model_button = PurpleButton(text="Upload model")
        self.upload_model_button.clicked.connect(self.upload_model)

        self.upload_model_metadata_button = PurpleButton(text="Upload metadata (keras)")
        self.upload_model_metadata_button.clicked.connect(self.upload_metadata)
        
        self.upload_model_layout.addWidget(self.upload_model_button)
        self.upload_model_layout.addWidget(self.upload_model_metadata_button)

        self.upload_weights_widget = QWidget()
        self.upload_weights_layout = QHBoxLayout(self.upload_weights_widget)
        self.upload_weights_button = PurpleButton(text="Upload weights")
        self.upload_weights_button.clicked.connect(self.upload_weights)

        self.framework_label = LabelWidget("Select Framework:")
        self.framework_dropdown = ComboBoxWidget()
        self.framework_dropdown.addItems(["Keras", "PyTorch", "StarDist"])
        self.framework_dropdown.currentIndexChanged.connect(self.update_model_dropdown)

        self.model_label = LabelWidget("Select Model:")
        self.model_dropdown = ComboBoxWidget()

        self.backbone_label = LabelWidget("Select Backbone:")
        self.backbone_dropdown = ComboBoxWidget()

        self.image_size_label = LabelWidget("Select image size:")
        self.image_size_dropdown = ComboBoxWidget()
        self.image_size_dropdown.addItems(["64x64", "128x128", "224x224", "256x256","512x512"])
        self.image_size_dropdown.setCurrentIndex(2)
        
        weight_form_layout = QFormLayout()
        weight_form_layout.addRow(self.framework_label, self.framework_dropdown)
        weight_form_layout.addRow(self.model_label, self.model_dropdown)
        weight_form_layout.addRow(self.backbone_label, self.backbone_dropdown)
        weight_form_layout.addRow(self.image_size_label,self.image_size_dropdown)

        self.upload_weights_layout.addWidget(self.upload_weights_button)
        self.upload_weights_layout.addLayout(weight_form_layout)

        self.select_pretrained_widget = QWidget()
        self.select_pretrained_layout = QHBoxLayout(self.select_pretrained_widget)
        self.pretrained_models_dropdown_label = LabelWidget("Select Pretrained model")
        self.pretrained_models_dropdown = ComboBoxWidget()
        self.pretrained_models_dropdown.addItems([
            "Mask R-CNN", "StarDist", "DeepCell", "Cellpose", "GAN"
        ])
        self.pretrained_models_dropdown.currentIndexChanged.connect(self.update_pretrained_model_dropdown)
        self.pretrained_models_dropdown.setCurrentIndex(0)
        self.select_pretrained_layout.addWidget(self.pretrained_models_dropdown_label)
        self.select_pretrained_layout.addWidget(self.pretrained_models_dropdown)

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
        form_layout.addRow(self.upload_model_widget)
        form_layout.addRow(self.upload_weights_widget)
        form_layout.addRow(self.select_pretrained_widget)
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

        self.analysis_metrics_widget = QWidget()
        self.analysis_metrics_widget.setMaximumWidth(200)
        self.analysis_metrics_layout = QVBoxLayout(self.analysis_metrics_widget)
        self.metrics_table = TableWidget(columns=["Id","Area"], min_width=100)
        self.export_button = PurpleButton(text="Export table")
        self.export_button.clicked.connect(self.export_table)
        self.analysis_metrics_layout.addWidget(self.metrics_table)
        self.analysis_metrics_layout.addWidget(self.export_button)
        splitter2 = SplitterWidget(Qt.Horizontal)
        splitter2.addWidget(splitter)
        splitter2.addWidget(self.analysis_metrics_widget)



        layout.addWidget(splitter2)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_pretrained_model_dropdown()
        self.update_method_dropdown()

    def export_table(self):
        from backend.data.excel_utils import export_to_excel
        export_to_excel(self.metrics_table.table)

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
        elif framework == "StarDist":
            self.model_dropdown.addItems(["StarDist"])
            self.backbone_dropdown.addItems(["StarDist"])

    def analysis_settings(self):
        from frontend.ui_analysisSettings import AnalysisSettingsDialog
        
        dialog = AnalysisSettingsDialog(self, first_visible_image=self.first_visible_image)
        if self.settings_values is not None:
            dialog.set_all_widget_values(self.settings_values)
        if dialog.exec_() == QDialog.Accepted:
            self.settings_values = dialog.get_all_widget_values()
            if self.settings_values is not None:
                self.controller.predictionController.save_settings(self.settings_values)

    def update_pretrained_model_dropdown(self):
        self.controller.predictionController.model_selected = self.pretrained_models_dropdown.currentText().lower()

    def update_method_dropdown(self):
        if self.method_dropdown.currentText() == "Uploaded model":
            self.upload_model_widget.setVisible(True)
            self.upload_weights_widget.setVisible(False)
            self.select_pretrained_widget.setVisible(False)
            self.controller.predictionController.predict_method = PredictMethods.UPLOADED_MODEL
        elif self.method_dropdown.currentText() == "Uploaded weights":
            self.update_model_dropdown()
            self.upload_model_widget.setVisible(False)
            self.upload_weights_widget.setVisible(True)
            self.select_pretrained_widget.setVisible(False)
            self.controller.predictionController.predict_method = PredictMethods.UPLOADED_WEIGHTS
        elif self.method_dropdown.currentText() == "Pretrained model":
            self.upload_model_widget.setVisible(False)
            self.upload_weights_widget.setVisible(False)
            self.select_pretrained_widget.setVisible(True)
            self.controller.predictionController.predict_method = PredictMethods.SELECTED_MODEL

    def upload_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Model", "", "Model Files (*.h5 *.pt *.pth *.keras)")
        if file_path:
            print(f"Model uploaded: {file_path}")
            self.controller.predictionController.load_model(file_path)

    def upload_metadata(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Metadata", "", "Metadata File (*.json)")
        if file_path:
            print(f"Metadata uploaded: {file_path}")
            self.controller.predictionController.load_metadata(file_path)

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

    def upload_updated_weights_settings(self):
        self.controller.predictionController.weights_uploaded_model_settings.model_framework = self.framework_dropdown.currentText().lower()
        self.controller.predictionController.weights_uploaded_model_settings.model_backbone = self.backbone_dropdown.currentText().lower()
        from backend.backend_types import model_mapping
        model_str = self.model_dropdown.currentText()
        if model_str in model_mapping:
            self.controller.predictionController.weights_uploaded_model_settings.model_type = model_mapping[model_str]      

        self.controller.predictionController.evaluate()   

    def set_controller(self, controller):
        self.controller = controller
        self.predict_button.clicked.connect(self.upload_updated_weights_settings)
        self.image_slider.slider.valueChanged.connect(self.controller.predictionController.predict_move_preview)
        self.threshold_slider.slider.valueChanged.connect(self.controller.predictionController.threshold_change)

        self.left_button.pressed.connect(self.controller.predictionController.predict_start_navigate_left)
        self.left_button.released.connect(self.controller.predictionController.predict_stop_navigate)
        self.right_button.pressed.connect(self.controller.predictionController.predict_start_navigate_right)
        self.right_button.released.connect(self.controller.predictionController.predict_stop_navigate)
