from enum import Enum

class modelTypes(Enum):
    UNET = 1
    UNETPP = 2
    DEEPLABV3 = 3
    FNP = 4
    STARDIST = 5

class modelFrameworks(Enum):
    KERAS = 1
    PYTORCH = 2

class modelBackbones(Enum):
    RESNET34 = 1
    RESNET50 = 2
    EFFICIENTNET_B3 = 3

class PredictMethods(Enum):
    UPLOADED_MODEL = 1
    UPLOADED_WEIGHTS = 2
    SELECTED_MODEL = 3 

model_mapping = {
    "U-Net": modelTypes.UNET,
    "U-Net++": modelTypes.UNETPP,
    "DeepLabV3": modelTypes.DEEPLABV3,
    "FPN": modelTypes.FNP,
    "StarDist": modelTypes.STARDIST
}

framework_mapping = {
    "Keras": modelFrameworks.KERAS,
    "PyTorch": modelFrameworks.PYTORCH
}

KERAS = "keras"
PYTORCH = "pytorch"
STARDIST = "stardist"

backbone_mapping = {
    "Resnet34": modelBackbones.RESNET34,
    "Resnet50": modelBackbones.RESNET50,
    "EfficientNet-B3": modelBackbones.EFFICIENTNET_B3
}
reverse_backbone_mapping = {v: k for k, v in backbone_mapping.items()}