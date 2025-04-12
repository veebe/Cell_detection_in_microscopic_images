from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from frontend.ui_training import TrainingTab  
from frontend.widgets.tabs import TabWidget
from frontend.ui_analysis import AnalysisTab

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

        self.analysis_tab = AnalysisTab()
        self.tabs.addTab(self.analysis_tab, "Analysis")

    def set_controller(self, controller):
        self.controller = controller
        self.training_tab.set_controller(controller) 
        self.analysis_tab.set_controller(controller)