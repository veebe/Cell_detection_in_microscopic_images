from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np

from frontend.ui_main import MainUI
from frontend.widgets.popUpWidget import PopUpWidget
from backend.backend_predict import PredictionController
from backend.backend_train import TrainingController

class CellDetectionController:
    def __init__(self, ui):
        self.ui = ui
    
        self.predictionController = PredictionController(self.ui)
        self.trainingController = TrainingController(self.ui)
        
        self.ui.set_controller(self)