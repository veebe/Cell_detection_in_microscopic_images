from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def save_weights(self, path: str):
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
