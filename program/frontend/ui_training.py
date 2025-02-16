from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSizePolicy, QToolTip, QPushButton, QSplitter, QDialog
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
from frontend.widgets.label import LabelWidget

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()

        layout = QHBoxLayout(self)

        self.test_set_visible = True
        self.first_visible_image = None
        self.settings_values = None

        self.left_container = QWidget()
        self.left_layout = QVBoxLayout(self.left_container)
        #layout.addLayout(self.left_layout)
        
        drops_horizontal_layout = QHBoxLayout()
        self.left_layout.addLayout(drops_horizontal_layout)

        self.training_set_layout = QVBoxLayout()
        drops_horizontal_layout.addLayout(self.training_set_layout)

        self.training_set_label = QLabel("Training set")
        self.training_set_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.training_set_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  
        self.training_set_layout.addWidget(self.training_set_label)

        self.test_mask_label = QLabel("Testing masks uploaded: 0")
        
        self.image_drop = DragDropWidget(self, "Drag & Drop Images Here", self.handle_drop)
        self.image_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_set_layout.addWidget(self.image_drop)
        self.mask_drop = DragDropWidget(self, "Drag & Drop Masks Here", self.handle_drop)
        self.mask_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.training_set_layout.addWidget(self.mask_drop)

        self.add_testing_set_button = IconButtonWidget("icons/mathematics-sign-minus-round-icon.svg")
        self.add_testing_set_button.setToolTip("Add a dedicated testing set")  
        self.add_testing_set_button.setFixedSize(20, 20)
        self.add_testing_set_button.setIconSize(QSize(20, 20))
        drops_horizontal_layout.addWidget(self.add_testing_set_button)

        self.testing_set_layout = QVBoxLayout()
        drops_horizontal_layout.addLayout(self.testing_set_layout)

        self.testing_set_label = QLabel("Testing set")
        self.testing_set_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.testing_set_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  
        self.testing_set_layout.addWidget(self.testing_set_label)
        
        self.test_image_drop = DragDropWidget(self, "Drag & Drop Images Here", self.handle_drop)
        self.test_image_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.testing_set_layout.addWidget(self.test_image_drop)
        self.test_mask_drop = DragDropWidget(self, "Drag & Drop Masks Here (not required)", self.handle_drop)
        self.test_mask_drop.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.testing_set_layout.addWidget(self.test_mask_drop)

        self.test_label_layout = QHBoxLayout()

        self.test_image_label = QLabel("Images uploaded: 0")
        self.test_image_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.test_image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  

        self.test_mask_label = QLabel("Masks uploaded: 0")
        self.test_mask_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.test_mask_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) 

        self.test_label_layout.addWidget(self.test_image_label)
        self.test_label_layout.addWidget(self.test_mask_label)
        self.testing_set_layout.addLayout(self.test_label_layout)

        self.label_layout = QHBoxLayout()

        self.image_label = QLabel("Images uploaded: 0")
        self.image_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  

        self.mask_label = QLabel("Masks uploaded: 0")
        self.mask_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.mask_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) 

        self.label_layout.addWidget(self.image_label)
        self.label_layout.addWidget(self.mask_label)
        self.training_set_layout.addLayout(self.label_layout)

        self.train_button = PurpleButton("Train and Detect")
        self.left_layout.addWidget(self.train_button)

        splitter = SplitterWidget(Qt.Horizontal)
        splitter2 = SplitterWidget(Qt.Horizontal)

        self.right_container = QWidget()
        self.middle_container = QWidget()
        splitter.addWidget(self.middle_container)
        splitter.addWidget(self.right_container)

        splitter2.addWidget(self.left_container)
        splitter2.addWidget(splitter)
        

        self.middle_layout = QVBoxLayout(self.middle_container)
        #layout.addLayout(self.middle_layout)

        model_settings_layout = QHBoxLayout()

        self.model_settings_button = IconButtonWidget("icons/settings-gear-icon.svg")
        self.model_settings_button.setToolTip("Model Settings")  
        self.model_settings_button.setFixedSize(20, 20)
        self.model_settings_button.setIconSize(QSize(20, 20))

        self.image_size_dropdown = ComboBoxWidget()
        self.image_size_dropdown.addItems(["64x64", "128x128", "256x256","512x512"])
        self.image_size_dropdown.setCurrentIndex(2)
        self.image_size_label = LabelWidget("Select training image size:")
        
        model_settings_layout.addWidget(self.image_size_label)
        model_settings_layout.addWidget(self.image_size_dropdown)
        model_settings_layout.addWidget(self.model_settings_button)

        self.middle_layout.addLayout(model_settings_layout)

        self.plot = PlotWidget()
        self.middle_layout.addWidget(self.plot)

        self.image_slider = SliderWidget()
        self.middle_layout.addWidget(self.image_slider)
 
        self.navigation_layout = QHBoxLayout()
        self.left_button = PurpleButton("<")
        self.navigation_layout.addWidget(self.left_button)
        self.right_button = PurpleButton(">")
        self.navigation_layout.addWidget(self.right_button)
        self.middle_layout.addLayout(self.navigation_layout)
        
        self.right_layout = QVBoxLayout(self.right_container)

        self.metrics_table = TableWidget()
        self.right_layout.addWidget(self.metrics_table)

        self.progress_bar = ProgressBarWidget()
        self.progress_bar.setRange(0, 100)
        self.right_layout.addWidget(self.progress_bar)

        self.download_layout = QHBoxLayout()
        
        self.download_model_button = PurpleButton("Download Model")
        self.download_layout.addWidget(self.download_model_button)
        self.download_model_button.setEnabled(False)
        self.download_model_button.setToolTip("Downloads the trained model, so it can be used later or for different applications")
        
        self.download_weights_button = PurpleButton("Download Weights")
        self.download_layout.addWidget(self.download_weights_button)
        self.download_weights_button.setEnabled(False)
        self.download_weights_button.setToolTip("Downloads the model weights, so it can be used later or for different applications")  
        
        self.right_layout.addLayout(self.download_layout)

        layout.addWidget(splitter2)

    def handle_drop(self, files, widget=None):
        if self.controller:
            if widget == self.image_drop:
                self.first_visible_image = files[0]
                self.controller.load_images(files)
            elif widget == self.mask_drop:
                self.controller.load_masks(files)
            elif widget == self.test_image_drop:
                self.controller.load_test_images(files)
            elif widget == self.test_mask_drop:
                self.controller.load_test_masks(files)

    def toggle_training_set(self):
        for i in range(self.testing_set_layout.count()):
            item = self.testing_set_layout.itemAt(i)
            if item and item.widget():
                widget = item.widget()
                widget.setVisible(not widget.isVisible())
            if item and item.layout():
                for j in range(item.count()):
                    layout_item = item.itemAt(j)
                    if layout_item and layout_item.widget():
                        widget = layout_item.widget()
                        widget.setVisible(not widget.isVisible())

        self.test_set_visible = not self.test_set_visible
        if not self.test_set_visible:
            self.add_testing_set_button.updateSVG("icons/mathematics-sign-plus-round-icon.svg")
            self.add_testing_set_button.setToolTip("Add a dedicated testing set")  
        else:
            self.add_testing_set_button.updateSVG("icons/mathematics-sign-minus-round-icon.svg")
            self.add_testing_set_button.setToolTip("Remove a dedicated testing set")  

    def model_settings(self):
        from frontend.ui_modelsettings import ModelSettingsDialog
        
        dialog = ModelSettingsDialog(self, first_visible_image=self.first_visible_image)
        if self.settings_values is not None:
            dialog.set_all_widget_values(self.settings_values)
        if dialog.exec_() == QDialog.Accepted:
            self.settings_values = dialog.get_all_widget_values()
        self.controller.save_settings(self.settings_values)
            
    def set_controller(self, controller):
        self.controller = controller
        self.train_button.clicked.connect(self.controller.train_networks)

        self.left_button.pressed.connect(self.controller.start_navigate_left)
        self.left_button.released.connect(self.controller.stop_navigate)
        self.right_button.pressed.connect(self.controller.start_navigate_right)
        self.right_button.released.connect(self.controller.stop_navigate)

        self.add_testing_set_button.clicked.connect(self.toggle_training_set)
        self.add_testing_set_button.clicked.connect(controller.toggle_training_set)
        self.model_settings_button.clicked.connect(self.model_settings)
        self.download_model_button.clicked.connect(self.controller.download_model)
        self.download_weights_button.clicked.connect(self.controller.download_weights)
        self.image_slider.slider.valueChanged.connect(self.controller.move_preview)
