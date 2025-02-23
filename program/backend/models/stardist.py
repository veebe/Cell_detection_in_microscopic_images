import numpy as np
import torch
from csbdeep.utils import normalize
from stardist.models import Config2D, StarDist2D
from backend.models.model import BaseModel


class StarDistModel(BaseModel):
    def __init__(self, basedir="stardist_models", pretrained=True, input_size=(256, 256)):
        super(StarDistModel, self).__init__()
        self.basedir = basedir
        self.input_size = input_size

        if pretrained:
            self.model = StarDist2D.from_pretrained("2D_versatile_he")
        else:
            config = Config2D(
                patch_size=(input_size[0],input_size[1]), 
                n_rays=32,             
                grid=(2, 2),           
                n_channel_in=3,        
                use_gpu=True
            )
            self.model = StarDist2D(config, basedir=self.basedir)

        self.epochs = 10
        self.batch_size = 16

    def compile(self):
        pass  

    def _train(self, X_train, Y_train, X_val, Y_val, callbacks=None):
        from stardist.models.model2d import StarDistData2D

        data_kwargs = {
            'n_rays': self.model.config.n_rays,
            'grid': self.model.config.grid,
            'batch_size': self.batch_size,
            'patch_size': (self.input_size[0],self.input_size[1]),
            'length' : None
            #'shape_completion': self.model.config.shape_completion,
            #'foreground_prob': self.model.config.foreground_prob,
        }

        data_train = StarDistData2D(X_train, Y_train, **data_kwargs)
        data_val = StarDistData2D(X_val, Y_val, **data_kwargs) if X_val is not None else None

        self.model.keras_model.fit(
            data_train,
            validation_data=data_val,
            epochs=self.epochs,
            callbacks=callbacks 
        )

    def predict(self, image):
        img_norm = normalize(image)
        labels, details = self.model.predict_instances(img_norm)
        return labels, details

    def save(self, path: str):
        self.model.export_TF(path)

    def save_weights(self, path: str):
        pass

    @classmethod
    def load(cls, path: str):
        return cls(basedir=path, pretrained=False)  
