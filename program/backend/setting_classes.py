from backend.backend_types import modelTypes


class ModelSettings:
    def __init__(self):
        self.model_type = modelTypes.UNETPP
        self.model_framework = "keras"
        self.model_backbone = "resnet34"
        self.epochs = 10
        self.val_split = 20
        self.batch = 16

    def save_settings(self, values):
        from backend.backend_types import model_mapping
        if 'framework' in values:
            self.model_framework = str(values['framework']).lower()

        if 'model' in values:
            model_str = values['model']
            if model_str in model_mapping:
                self.model_type = model_mapping[model_str]

        if 'backbone' in values:
            self.model_backbone = str(values['backbone']).lower()

        if 'epochs' in values:
            self.epochs = int(values['epochs'])

        if 'batch' in values:
            self.batch = int(2 ** values['batch'] )
        
        if 'validation_split' in values:
            self.val_split = int(values['validation_split'])

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

    def save_settings(self, values):
        if 'gaussian_blur' in values:
            gaussian = values['gaussian_blur']
            if isinstance(gaussian, dict) and 'enabled' in gaussian and 'value' in gaussian:
                self.blur_check = gaussian['enabled']
                self.blur = int(gaussian['value'])

        if 'brightness' in values:
            brightness = values['brightness']
            if isinstance(brightness, dict) and 'enabled' in brightness and 'value' in brightness:
                self.brightness_check = brightness['enabled']
                self.brightness = int(brightness['value'])

        if 'contrast' in values:
            contrast = values['contrast']
            if isinstance(contrast, dict) and 'enabled' in contrast and 'value' in contrast:
                self.contrast_check = contrast['enabled']
                self.contrast = int(contrast['value'])

        if 'denoise' in values:
            denoise = values['denoise']
            if isinstance(denoise, dict) and 'enabled' in denoise and 'value' in denoise:
                self.denoise_check = denoise['enabled']
                self.denoise = int(denoise['value'])