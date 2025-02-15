from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QTabWidget, QApplication
from frontend.ui_training import TrainingTab  
from PyQt5.QtCore import Qt
from frontend.widgets.tabs import TabWidget

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

        self.tabs = TabWidget()

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

    def set_controller(self, controller):
        self.controller = controller
        self.training_tab.set_controller(controller) 
