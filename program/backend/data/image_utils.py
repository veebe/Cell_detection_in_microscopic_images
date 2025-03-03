import cv2
from backend.setting_classes import PreprocessingSettings

def cv2_images_preprocess(cv_images, preprocess_settings : PreprocessingSettings):
    images = []
    for img in cv_images:
        if preprocess_settings.denoise_check:
            h = preprocess_settings.denoise
            if len(img.shape) == 3:
                img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
            else:  
                img = cv2.fastNlMeansDenoising(img, None, h)

        if preprocess_settings.blur_check:
            ksize = preprocess_settings.blur
            ksize = ksize + 1 if ksize % 2 == 0 else ksize 
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        if preprocess_settings.contrast_check:
            alpha = preprocess_settings.contrast / 100
            img = cv2.convertScaleAbs(img, alpha=alpha)
        if preprocess_settings.brightness_check:
            beta = preprocess_settings.brightness
            img = cv2.convertScaleAbs(img, beta=beta)

        images.append(img)
    return images

def paths_to_cv2_images(paths, color_mode = cv2.COLOR_BGR2RGB):
    images = []
    for path in paths:
        img = cv2.imread(path)  
        if img is not None:
            img = cv2.cvtColor(img, color_mode)
            images.append(img)
    return images

def cv2_images_resize(cv_images, image_width, image_height):
    images = []
    for img in cv_images:
        img = cv2.resize(img,(image_width, image_height))
        images.append(img)
    return images

def backbone_preprocess(np_images,framework="",backbone=""):
    if backbone != "":
        from backend.backend_types import PYTORCH, KERAS
        if framework == PYTORCH:
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            preprocess_input = get_preprocessing_fn(encoder_name=backbone, pretrained='imagenet')
        elif framework == KERAS:
            import segmentation_models as sm
            preprocess_input = sm.get_preprocessing(backbone)
        else:
            return np_images
        return preprocess_input(np_images)