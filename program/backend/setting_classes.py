from backend.backend_types import modelFrameworks, modelBackbones, modelTypes


class ModelSettings:
    def __init__(self):
        self.model_type = modelTypes.UNETPP
        self.model_framework = "keras"
        self.model_backbone = "resnet34"
        self.epochs = 10
        self.val_split = 20
        self.batch = 16

class PreprocessingSettings:
    def __init__(self):
        self.blur_check = False
        self.blur = 0
        self.brightness_check = False
        self.brightness = 0
        self.contrast_check = False
        self.contrast = 100
        self.denoise_check = False
        self.denoise = 0