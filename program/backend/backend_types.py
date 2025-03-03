from enum import Enum
import cv2

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
UNKNOWN = ""

backbone_mapping = {
    "Resnet34": modelBackbones.RESNET34,
    "Resnet50": modelBackbones.RESNET50,
    "EfficientNet-B3": modelBackbones.EFFICIENTNET_B3
}
reverse_backbone_mapping = {v: k for k, v in backbone_mapping.items()}

colormap_dict = {
            "AUTUMN": cv2.COLORMAP_AUTUMN,
            "BONE": cv2.COLORMAP_BONE,
            "JET": cv2.COLORMAP_JET,
            "WINTER": cv2.COLORMAP_WINTER,
            "RAINBOW": cv2.COLORMAP_RAINBOW,
            "OCEAN": cv2.COLORMAP_OCEAN,
            "SUMMER": cv2.COLORMAP_SUMMER,
            "SPRING": cv2.COLORMAP_SPRING,
            "COOL": cv2.COLORMAP_COOL,
            "HSV": cv2.COLORMAP_HSV,
            "PINK": cv2.COLORMAP_PINK,
            "HOT": cv2.COLORMAP_HOT,
            "PARULA": cv2.COLORMAP_PARULA,
            "MAGMA": cv2.COLORMAP_MAGMA,
            "INFERNO": cv2.COLORMAP_INFERNO,
            "PLASMA": cv2.COLORMAP_PLASMA,
            "VIRIDIS": cv2.COLORMAP_VIRIDIS,
            "CIVIDIS": cv2.COLORMAP_CIVIDIS,
            "TWILIGHT": cv2.COLORMAP_TWILIGHT,
            "TWILIGHT SHIFTED": cv2.COLORMAP_TWILIGHT_SHIFTED,
            "TURBO": cv2.COLORMAP_TURBO,
            "DEEPGREEN": cv2.COLORMAP_DEEPGREEN
        }