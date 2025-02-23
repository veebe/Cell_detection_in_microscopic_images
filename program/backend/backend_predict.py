import cv2
from PyQt5.QtWidgets import QTableWidgetItem
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import numpy as np
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
import yolov5 as yol
from PyQt5.QtWidgets import QFileDialog

from backend.setting_classes import PreprocessingSettings, ModelSettings
from backend.backend_types import PredictMethods

class PredictionController:
    def __init__(self, ui):
        self.model_uploaded = None
        self.model_selected = None
        self.weights_uploaded = None
        self.weights_uploaded_path = ""
        self.model_uploaded_path = ""
        self.metadata_uploaded_path = ""
        self.weights_uploaded_model_settings = ModelSettings()

        self.eval_image_paths = []
        self.eval_images = []
        self.eval_images_preprocessed = []
        self.eval_images_preprocessed_np = []
        self.probabilities = []
        self.threshold = 50

        self.preprocess_settings = PreprocessingSettings()
        self.predict_method = PredictMethods.UPLOADED_MODEL
        self.image_width = 0
        self.image_height = 0 

        self.model = None
        self.framework = ""
        self.preview_image_index = 0
        self.predicitons = []

        self.left_timer = QTimer()
        self.right_timer = QTimer()
        self.left_timer.timeout.connect(self.predict_navigate_left)
        self.right_timer.timeout.connect(self.predict_navigate_right)

        self.ui = ui
        self.device = "cpu"
        self.model_name = ""

    def paths_to_cv2_images(self, paths):
        images = []
        for path in paths:
            img = cv2.imread(path)  
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
        return images

    def cv2_images_preprocess(self,cv_images):
        images = []
        for img in cv_images:
            if self.preprocess_settings.denoise_check:
                h = self.preprocess_settings.denoise
                if len(img.shape) == 3:
                    img = cv2.fastNlMeansDenoisingColored(img, None, h, h)
                else:  
                    img = cv2.fastNlMeansDenoising(img, None, h)

            if self.preprocess_settings.blur_check:
                ksize = self.preprocess_settings.blur
                ksize = ksize + 1 if ksize % 2 == 0 else ksize 
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

            if self.preprocess_settings.contrast_check:
                alpha = self.preprocess_settings.contrast / 100
                img = cv2.convertScaleAbs(img, alpha=alpha)
            if self.preprocess_settings.brightness_check:
                beta = self.preprocess_settings.brightness
                img = cv2.convertScaleAbs(img, beta=beta)

            images.append(img)
        return images

    def cv2_images_resize(self, cv_images):
        images = []
        for img in cv_images:
            if self.model_name in ["stardist", "cellpose"]:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img = cv2.resize(img,(self.image_width, self.image_height))
            images.append(img)
        return images
        
    def backbone_preprocess(self, np_images):
        if self.framework == ".pth" or self.framework == ".pt":
            from segmentation_models_pytorch.encoders import get_preprocessing_fn
            if hasattr(self.model, "backbone"):
                preprocess_input = get_preprocessing_fn(
                    encoder_name=self.model.backbone,
                    pretrained='imagenet'
                )
                return preprocess_input(np_images)
        else:
            import segmentation_models as sm
            preprocess_input = sm.get_preprocessing(self.model.backbone)
            return preprocess_input(np_images)

    def load_model(self, path):
        import os
        self.framework = os.path.splitext(path)[1]
        self.model_uploaded_path = path

    def load_weights(self,path):
        import os
        self.framework = os.path.splitext(path)[1]
        self.weights_uploaded_path = path

    def load_metadata(self,path):
        self.metadata_uploaded_path = path



    def load_pretrained_model(self, model_name):
        model_name = model_name.lower()
        try:
            if model_name == "mask_rcnn":
                from detectron2 import model_zoo
                from detectron2.config import get_cfg

                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
                cfg.MODEL.DEVICE = self.device
                model = cfg  
                model.input_size = (cfg.INPUT.MIN_SIZE_TRAIN[0], cfg.INPUT.MIN_SIZE_TRAIN[1])
                model.backbone = "resnet50"  
                return model

            elif model_name == "stardist": 
                from stardist.models import StarDist2D
                model = StarDist2D.from_pretrained('2D_versatile_he')
                model.input_size = (256, 256)    
                return model

            elif model_name == "deepcell":
                from deepcell.applications import Mesmer
                model = Mesmer()
                model.input_size = (256, 256)
                return model

            elif model_name == "cellpose":
                from cellpose import models
                model = models.Cellpose(gpu=(self.device == "cuda"), model_type='cyto')
                model.input_size = (256, 256)  
                return model

            elif model_name == "gan":
                import torch
                from segmentation_models_pytorch import Unet
                model = Unet(encoder_name="resnet34", classes=1)
                model.input_size = (256, 256)
                model.backbone = "resnet34"  
                return model

            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            return None

    def evaluate(self):
        import torch
        if self.predict_method == PredictMethods.SELECTED_MODEL:
            print(self.model_selected)
            self.model = self.load_pretrained_model(self.model_selected)
            self.framework = ".pth"
        elif self.predict_method == PredictMethods.UPLOADED_MODEL:
            if self.framework == ".pth" or self.framework == ".pt":
                from backend.models.pytorch import PyTorchModel
                self.model_uploaded = PyTorchModel.load(path=self.model_uploaded_path)
            else:
                from backend.models.keras import KerasModel
                self.model_uploaded = KerasModel.load(path=self.model_uploaded_path,meta_path=self.metadata_uploaded_path)
            self.model = self.model_uploaded
        elif self.predict_method == PredictMethods.UPLOADED_WEIGHTS:
            print("ulpading")
            print(f"{self.weights_uploaded_model_settings.model_type} {self.weights_uploaded_model_settings.model_backbone} {self.weights_uploaded_model_settings.model_framework}")

            self.image_height = int(self.ui.analysis_tab.image_size_dropdown.currentText().split('x')[0]) 
            self.image_width = int(self.ui.analysis_tab.image_size_dropdown.currentText().split('x')[0]) 
            if self.weights_uploaded_path == "":
                popup = PopUpWidget("error","No path")
                popup.show()
                return
            if self.weights_uploaded_model_settings.model_framework == "pytorch":
                from backend.models.pytorch import PyTorchModel
                self.weights_uploaded = PyTorchModel(self.weights_uploaded_model_settings.model_type,self.weights_uploaded_model_settings.model_backbone, input_size=(self.image_height,self.image_width))
                self.weights_uploaded.load_weights(path=self.weights_uploaded_path)
                self.framework = ".pth"
            elif self.weights_uploaded_model_settings.model_framework == "keras":
                from backend.models.keras import KerasModel
                self.weights_uploaded = KerasModel(self.weights_uploaded_model_settings.model_backbone,input_size=(self.image_height,self.image_width,3))
            self.model = self.weights_uploaded

        if self.model == None:
            popup = PopUpWidget("error", "No model")
            popup.show()
            return
        if len(self.eval_image_paths) == 0:
            popup = PopUpWidget("error", "No Images")
            popup.show()
            return

        self.image_width = self.model.input_size[0]
        self.image_height = self.model.input_size[1]

        self.eval_images = self.paths_to_cv2_images(self.eval_image_paths)
        self.eval_images_preprocessed = self.cv2_images_preprocess(self.eval_images)
        self.eval_images_preprocessed = self.cv2_images_resize(self.eval_images_preprocessed)
        self.eval_images_preprocessed_np = np.array(self.eval_images_preprocessed)
        self.eval_images_preprocessed_np = self.backbone_preprocess(self.eval_images_preprocessed_np)
        
        if self.framework == ".pth" or self.framework == ".pt":
            #from deepcell.applications import Mesmer
            from stardist.models import StarDist2D
            from cellpose import models
            if isinstance(self.model, StarDist2D):
                from skimage.exposure import rescale_intensity
                from skimage.color import label2rgb
                from skimage.io import imread

                image = imread(self.eval_image_paths[0])

                from skimage.exposure import rescale_intensity

                image_normed = rescale_intensity(image, out_range=(0, 1))

                print(f'Intensity range: [{image_normed.min()} - {image_normed.max()}]')
                print(f'Array type: {image_normed.dtype}')

                labels, polys = self.model.predict_instances(
                    image_normed,
                    axes="YXC",
                    prob_thresh=0.5,  
                    nms_thresh=0.1, 
                    scale=1, 
                    return_labels=True,
                )

                probabilities = list(polys["prob"])

                n_detections = len(probabilities)

                print(f'{n_detections} cells detected.')

                from skimage.color import label2rgb
                rgb_composite = label2rgb(labels, image=image, bg_label=0)

                self.ui.analysis_tab.segmented_image.display_image(rgb_composite)


                return()

            #elif isinstance(self.model, Mesmer):
            #    masks = self.model.predict(np.expand_dims(self.eval_images_preprocessed_np, -1))
            #    binary_predictions = masks[...,0] * 255

            elif isinstance(self.model, models.Cellpose):
                masks, _, _ = self.model.eval(self.eval_images_preprocessed_np, diameter=None)
                binary_predictions = (masks > 0).astype(np.uint8) * 255
            
            #elif "mask_rcnn" in str(type(self.model)):
            #    # Mask R-CNN prediction
            #    from detectron2.engine import DefaultPredictor
            #    predictor = DefaultPredictor(self.model)
            #    outputs = predictor(self.eval_images_preprocessed_np[0])
            #    masks = outputs["instances"].pred_masks.cpu().numpy()
            #    binary_predictions = np.any(masks, axis=0).astype(np.uint8) * 255
            
            else:
                import torch
                self.X_val_predict = torch.from_numpy(self.eval_images_preprocessed_np).float()
                print(self.X_val_predict.shape)
                print(self.eval_images_preprocessed_np.shape)
                self.X_val_predict = self.X_val_predict.permute(0, 3, 1, 2).to(self.model.device)
                
                if self.X_val_predict.max() > 1.0:
                    self.X_val_predict = self.X_val_predict / 255.0
                
                pad_h = (32 - self.X_val_predict.shape[2] % 32) % 32
                pad_w = (32 - self.X_val_predict.shape[3] % 32) % 32
                if pad_h > 0 or pad_w > 0:
                    self.X_val_predict = torch.nn.functional.pad(self.X_val_predict, (0, pad_w, 0, pad_h))
                
                batch_size = 4 
                predictions = []
                for i in range(0, len(self.X_val_predict), batch_size):
                    batch = self.X_val_predict[i:i+batch_size]
                    with torch.no_grad():
                        pred = self.model.predict(batch)
                        predictions.append(pred)
                predictions = torch.cat(predictions)
                self.probabilities = torch.sigmoid(predictions)
                #binary_predictions = (self.probabilities > (self.threshold / 100)).squeeze(1).cpu().numpy()
                
                #binary_predictions = (binary_predictions * 255).astype(np.uint8)
        
        else:
            predictions = self.model.predict(self.eval_images_preprocessed_np)
            
            if predictions.ndim == 4:
                self.probabilities = predictions.squeeze(-1)
            else:
                self.probabilities = predictions.copy()
            #binary_predictions = (self.probabilities > (self.threshold/100)).astype(np.uint8) * 255
        
        self.threshold_change(self.threshold)

        """
        self.predicitons = binary_predictions
        
        self.postprocess_cells(binary_predictions)
        self.disply_current_predicted_image()
        return binary_predictions
        """

    def resize_to_original(self, segmented_images):
        resized_images = []
        for i, seg_img in enumerate(segmented_images):
            original_height, original_width = self.eval_images[i].shape[:2] 
            resized_img = cv2.resize(seg_img, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            resized_images.append(resized_img)
        return resized_images

    def threshold_change(self, value):
        self.threshold = value
        if len(self.probabilities) > 0:
            if self.framework in [".pth", ".pt"]: 
                binary_predictions = (self.probabilities > (value/100)).squeeze(1).cpu().numpy()
            else:  
                binary_predictions = (self.probabilities > (value/100))
            
            binary_predictions = (binary_predictions * 255).astype(np.uint8)
            
            self.predicitons = self.resize_to_original(binary_predictions)
            self.postprocess_cells(binary_predictions)
            self.disply_current_predicted_image()
      
    def separate_cells(self, predictions):
        processed_predictions = []
        
        for i in range(len(predictions)):
            binary = predictions[i].astype(np.uint8)
            
            #kernel = np.ones((3,3), np.uint8)
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
            
            if np.max(output) == 0:
                print(f"No cells detected in image {i}, using original mask")
                output = binary
            
            processed_predictions.append(output)
        
        processed_predictions = np.array(processed_predictions)
        self.processed_predictions = self.resize_to_original(processed_predictions)

    def postprocess_cells(self, predictions):

        if self.model_name == "mask_rcnn":
            predictions = np.stack([np.any(m, axis=0) for m in predictions])
        elif self.model_name == "stardist":
            predictions = (predictions > 0).astype(np.uint8)

        try:
            kernel = np.ones((3, 3), np.uint8)
            cleaned_predictions = []
            
            for pred in predictions:
                binary = pred.astype(np.uint8)
                #binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
                #binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
                cleaned_predictions.append(binary)
            
            cleaned_predictions = np.array(cleaned_predictions)
            
            self.separate_cells(cleaned_predictions)
            
        
        except Exception as e:
            print(f"Error in postprocess_cells: {str(e)}")
            self.processed_predictions = predictions

    def disply_current_predicted_image(self):
        self.ui.analysis_tab.eval_images_drop.display_image(self.eval_image_paths[self.preview_image_index])
        if len(self.predicitons) > 0:
            self.ui.analysis_tab.predicted_image.display_image(self.predicitons[self.preview_image_index])

            current_mask = self.processed_predictions[self.preview_image_index]
            current_image = self.label_segments(current_mask)
            self.ui.analysis_tab.segmented_image.display_image(current_image)

    def label_segments(self, current_mask):
            colored_mask = cv2.applyColorMap(current_mask, cv2.COLORMAP_JET)

            label_overlay = np.zeros_like(colored_mask)

            unique_labels = np.unique(current_mask)
            unique_labels = unique_labels[unique_labels > 0]  

            self.ui.analysis_tab.metrics_table.clear_table()           
            id = 1
            for label_id in unique_labels:
                cell_mask = (current_mask == label_id).astype(np.uint8)

                contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if len(contours) > 0:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest_contour)
                    M = cv2.moments(largest_contour)

                    if M["m00"] > 0:  
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])

                        self.ui.analysis_tab.metrics_table.add_row([id,area])

                        cv2.putText(label_overlay, str(id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8, cv2.LINE_AA)
                        id += 1

            return cv2.addWeighted(colored_mask, 0.8, label_overlay, 1.0, 0)

    def predict_start_navigate_left(self):
        self.predict_navigate_left()  
        self.left_timer.start(100)  

    def predict_start_navigate_right(self):
        self.predict_navigate_right()  
        self.right_timer.start(100) 

    def predict_stop_navigate(self):
        self.left_timer.stop()
        self.right_timer.stop()

    def predict_navigate_left(self):
        if self.preview_image_index > 0:
            self.preview_image_index -= 1
            self.disply_current_predicted_image()
            self.ui.analysis_tab.image_slider.slider.setValue(self.preview_image_index)

    def predict_navigate_right(self):
        if self.preview_image_index < len(self.eval_image_paths) - 1:
            self.preview_image_index += 1
            self.disply_current_predicted_image()
            self.ui.analysis_tab.image_slider.slider.setValue(self.preview_image_index)
    
    def predict_move_preview(self, value):
        self.preview_image_index = value
        self.disply_current_predicted_image()
 
    def save_settings(self, values):
        if 'gaussian_blur' in values:
            gaussian = values['gaussian_blur']
            if isinstance(gaussian, dict) and 'enabled' in gaussian and 'value' in gaussian:
                self.preprocess_settings.blur_check = gaussian['enabled']
                self.preprocess_settings.blur = int(gaussian['value'])

        if 'brightness' in values:
            brightness = values['brightness']
            if isinstance(brightness, dict) and 'enabled' in brightness and 'value' in brightness:
                self.preprocess_settings.brightness_check = brightness['enabled']
                self.preprocess_settings.brightness = int(brightness['value'])

        if 'contrast' in values:
            contrast = values['contrast']
            if isinstance(contrast, dict) and 'enabled' in contrast and 'value' in contrast:
                self.preprocess_settings.contrast_check = contrast['enabled']
                self.preprocess_settings.contrast = int(contrast['value'])

        if 'denoise' in values:
            denoise = values['denoise']
            if isinstance(denoise, dict) and 'enabled' in denoise and 'value' in denoise:
                self.preprocess_settings.denoise_check = denoise['enabled']
                self.preprocess_settings.denoise = int(denoise['value'])

        print("Preprocessing Settings Updated:", self.preprocess_settings.__dict__)