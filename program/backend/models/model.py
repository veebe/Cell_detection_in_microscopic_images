from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load(self, path: str):
        pass

    def save(self, path: str):
        from frontend.widgets.popUpWidget import PopUpWidget
        popup = PopUpWidget("info", f"Model saved to: {path}")
        popup.show()
        pass

    def save_weights(self, path: str):
        from frontend.widgets.popUpWidget import PopUpWidget
        popup = PopUpWidget("info", f"Model weights saved to: {path}")
        popup.show()
        pass

    @abstractmethod
    def compile(self, **kwargs):
        pass

    @abstractmethod
    def _train(self, **kwargs):
        pass

    @abstractmethod
    def predict(self, inputs):
        pass
