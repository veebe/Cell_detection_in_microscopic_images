from backend.backend_predict import PredictionController
from backend.backend_train import TrainingController

class CellDetectionController:
    def __init__(self, ui):
        self.ui = ui
    
        self.predictionController = PredictionController(self.ui)
        self.trainingController = TrainingController(self.ui)
        
        self.ui.set_controller(self)