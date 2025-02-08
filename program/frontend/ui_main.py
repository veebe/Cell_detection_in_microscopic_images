from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QTabWidget, QSizePolicy, QProgressBar
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from frontend.widgets.button import PurpleButton
from frontend.widgets.dragDrop import DragDropWidget
from frontend.widgets.plot import PlotWidget
from frontend.widgets.table import TableWidget
from frontend.widgets.processBar import ProgressBarWidget

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Detection Application")
        self.setGeometry(100, 100, 1000, 600)

        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        self.setStyleSheet("background-color: #3C3737;")

        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                background-color: #212020; 
                border: 3px solid #803eb5;
                border-top-right-radius: 5px;
                border-bottom-left-radius: 5px;
                border-bottom-right-radius: 5px;
            }
            QTabBar::tab { 
                background: #591d8a; 
                padding: 8px;
                font-weight: bold;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                min-width: 120px; 
                color: #bfbfbf;
            }
            QTabBar::tab:selected { 
                background: #803eb5;  
                color: white;
            }
            QTabBar::tab:hover { 
                background: #993399;  
                color: #ffffff;
            }
        """)

        self.main_layout.addWidget(self.tabs)

        self.training_tab = QWidget()
        self.create_training_tab()
        self.tabs.addTab(self.training_tab, "Training")

        self.analysis_tab = QWidget()
        self.create_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")


    def create_training_tab(self):
        layout = QHBoxLayout(self.training_tab)

        self.left_layout = QVBoxLayout()
        layout.addLayout(self.left_layout)

        self.image_drop = DragDropWidget(self, "Drag & Drop Images Here", self.handle_drop)
        self.image_drop.setMinimumWidth(300)
        self.left_layout.addWidget(self.image_drop)

        self.mask_drop = DragDropWidget(self, "Drag & Drop Masks Here", self.handle_drop)
        self.mask_drop.setMinimumWidth(300)
        self.left_layout.addWidget(self.mask_drop)

        self.label_layout = QHBoxLayout()

        self.image_label = QLabel("Images uploaded: 0")
        self.image_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)  

        self.mask_label = QLabel("Masks uploaded: 0")
        self.mask_label.setStyleSheet("color:#ffffff; font-size: 9px; font-weight: bold;")
        self.mask_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed) 

        self.label_layout.addWidget(self.image_label)
        self.label_layout.addWidget(self.mask_label)
        self.left_layout.addLayout(self.label_layout)

        self.train_button = PurpleButton("Train and Detect")
        self.left_layout.addWidget(self.train_button)

        self.middle_layout = QVBoxLayout()
        layout.addLayout(self.middle_layout)

        self.plot = PlotWidget()
        self.middle_layout.addWidget(self.plot)

        self.navigation_layout = QHBoxLayout()
        self.left_button = PurpleButton("<")
        self.navigation_layout.addWidget(self.left_button)
        self.right_button = PurpleButton(">")
        self.navigation_layout.addWidget(self.right_button)
        self.middle_layout.addLayout(self.navigation_layout)

        self.right_layout = QVBoxLayout()
        layout.addLayout(self.right_layout)

        self.metrics_table = TableWidget()
        self.right_layout.addWidget(self.metrics_table)

        self.progress_bar = ProgressBarWidget()
        self.progress_bar.setRange(0, 100)
        self.right_layout.addWidget(self.progress_bar)

    def create_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)

        self.analysis_label = QLabel("Analysis tools will be added here.")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.analysis_label)

    def handle_drop(self, files, widget):
        if widget == self.image_drop:
            self.controller.load_images(files)
        elif widget == self.mask_drop:
            self.controller.load_masks(files)

    def update_display(self):
        self.controller.display_current_image()

    def set_controller(self, controller):
        self.controller = controller
        self.train_button.clicked.connect(self.controller.train_networks)
        self.left_button.clicked.connect(self.controller.navigate_left)
        self.right_button.clicked.connect(self.controller.navigate_right)
        #self.slider.valueChanged.connect(self.controller.change_train_test_distribution)
