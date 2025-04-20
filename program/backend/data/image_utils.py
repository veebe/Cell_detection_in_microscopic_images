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
    

def apply_standard_watershed(binary):
    import cv2
    import numpy as np

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    markers = cv2.watershed(binary_bgr, markers)
    
    output = np.zeros_like(binary)
    
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            if area > 30:  
                output[markers == label] = label * (255 // markers.max())
    
    return output

def apply_marker_based_watershed(binary):
    import cv2
    import numpy as np
    
    binary = binary.astype(np.uint8)
    
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(binary, kernel, iterations=3)
    
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    dist_max = dist_transform.max()
    thresh_val = 0.3 * dist_max 
    
    _, sure_fg = cv2.threshold(dist_transform, thresh_val, 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    _, markers = cv2.connectedComponents(sure_fg)
    
    markers = markers + 1
    
    markers[unknown == 255] = 0
    
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    output = np.zeros_like(binary)
    
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype(np.uint8)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            if area > 20: 
                output[markers == label] = label * (255 // markers.max())
    
    return output

def apply_distance_transform_watershed(binary):
    import cv2
    import numpy as np
    from skimage.feature import peak_local_max
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed
    
    binary = binary.astype(np.uint8)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    dist_normalized = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    local_max = peak_local_max(dist, min_distance=10, labels=binary, footprint=np.ones((3, 3)), exclude_border=False)
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (r, c) in enumerate(local_max):
        markers[r, c] = i + 1

    labels = watershed(-dist, markers, mask=binary)

    output = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, labels.max() + 1):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours and cv2.contourArea(contours[0]) > 30:
            output[labels == label] = label * (255 // labels.max())
    
    return output

def apply_h_minima_watershed(binary):
    import cv2
    import numpy as np
    from skimage.morphology import reconstruction
    
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    inverted_dist = 255 - dist_normalized
    
    h = 0.23 * np.max(inverted_dist)
    seed = inverted_dist.copy().astype(np.float64) + h
    mask = inverted_dist.astype(np.float64)
    recon = reconstruction(seed, mask, method='erosion')
    h_minima = recon - inverted_dist
    
    _, markers = cv2.threshold(h_minima.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    markers = markers.astype(np.uint8)
    
    _, markers = cv2.connectedComponents(markers)
    markers = markers + 1
    markers[binary == 0] = 0
    
    img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_color, markers)
    
    output = np.zeros_like(binary)
    for label in range(2, markers.max() + 1):
        mask = (markers == label).astype(np.uint8)
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        mask = cv2.bitwise_and(mask, binary)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0 and cv2.contourArea(contours[0]) > 30:
            output[mask > 0] = label * (255 // markers.max())
        
    return output

def apply_compact_watershed(binary):
    import cv2
    import numpy as np
    from skimage.segmentation import watershed as sk_watershed
    
    
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(dist_normalized, kernel)
    local_max = (dist_normalized == dilated) & (dist_normalized > 30)
    local_max = local_max.astype(np.uint8) * 255
    
    num_labels, labels = cv2.connectedComponents(local_max)
    
    inverted_dist = 255 - dist_normalized
    
    labels = sk_watershed(inverted_dist, labels, mask=binary, compactness=0.5)
    
    output = np.zeros_like(binary)
    for label in range(1, labels.max() + 1):
        mask = (labels == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            area = cv2.contourArea(contours[0])
            if area > 30:
                output[labels == label] = label * (255 // labels.max())
    
    return output