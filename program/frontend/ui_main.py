from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QApplication
from frontend.ui_training import TrainingTab  
from PyQt5.QtCore import Qt

class MainUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cell Detection Application")
        self.setGeometry(100, 100, 1500, 600)

        QApplication.instance().setStyleSheet("""
            QToolTip {
                color: white;
                background-color: black;
                border: 1px solid white;
                padding: 5px;
                font-size: 12px;
            }
        """)

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

        self.training_tab = TrainingTab()
        self.tabs.addTab(self.training_tab, "Training")

        self.analysis_tab = QWidget()
        self.create_analysis_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

    def create_analysis_tab(self):
        layout = QVBoxLayout(self.analysis_tab)

        self.analysis_label = QLabel("Analysis tools will be added here.")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.analysis_label)
    """
    def handle_drop(self, files, widget):
        if widget == self.training_tab.image_drop:
            self.controller.load_images(files)
        elif widget == self.training_tab.mask_drop:
            self.controller.load_masks(files)
    """

    def set_controller(self, controller):
        self.controller = controller
        self.training_tab.set_controller(controller) 
