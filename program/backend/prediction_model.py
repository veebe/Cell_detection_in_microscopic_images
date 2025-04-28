import cv2
import numpy as np
from frontend.widgets.popUpWidget import PopUpWidget
from backend.setting_classes import PreprocessingSettings
from backend.backend_types import PYTORCH, KERAS, UNKNOWN, WatershedAlgorithm

class PredictionModel:
    def __init__(self):
        self.model_uploaded = None
        self.model_selected = None
        self.weights_uploaded = None
        self.weights_uploaded_path = ""
        self.model_uploaded_path = ""
        self.metadata_uploaded_path = ""
        
        self.eval_image_paths = []
        self.eval_images = []
        self.eval_images_preprocessed = []
        self.eval_images_preprocessed_np = []
        self.probabilities = []
        self.threshold = 50
        self.current_color_option = cv2.COLORMAP_JET
        self.current_watershed_algorithm = WatershedAlgorithm.STANDARD
        
        self.preprocess_settings = PreprocessingSettings()
        self.image_width = 0
        self.image_height = 0 
        
        self.model = None
        self.framework = ""
        self.preview_image_index = 0
        self.predictions = []
        self.binary_predictions = []
        self.processed_predictions = []
        
        self.device = "cpu"
        self.model_name = ""
    
    def load_model(self, path):
        self.parse_framework(path)
        self.model_uploaded_path = path

    def load_weights(self, path):
        self.parse_framework(path)
        self.weights_uploaded_path = path

    def parse_framework(self, path):
        import os
        framework = os.path.splitext(path)[1]
        if framework in [".pth", ".pt"]:
            self.framework = PYTORCH
        elif framework in [".keras", ".h5"]:
            self.framework = KERAS
        else:
            self.framework = UNKNOWN

    def load_metadata(self, path):
        self.metadata_uploaded_path = path

    def load_pretrained_model(self, model_name):
        self.framework = ""
        self.model_name = model_name.lower()
    
        try:
            if model_name == "stardist": 
                from stardist.models import StarDist2D
                model = StarDist2D.from_pretrained('2D_versatile_he')
                model.input_size = (512, 512)   
                model.backbone = "" 
                return model
            
            elif model_name == "cellpose":
                from cellpose import models
                model = models.Cellpose(gpu=(self.device == "cuda"), model_type='cyto')
                model.input_size = (512, 512)  
                return model

            else:
                raise ValueError(f"Unknown model: {model_name}")

        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            return None
    
    def uploaded_model_init(self):
        if self.framework == PYTORCH:
            from backend.models.pytorch import PyTorchModel
            self.model_uploaded = PyTorchModel.load(path=self.model_uploaded_path)
        elif self.framework == KERAS:
            from backend.models.keras import KerasModel
            self.model_uploaded = KerasModel.load(path=self.model_uploaded_path, meta_path=self.metadata_uploaded_path)
        self.model = self.model_uploaded

    def evaluate_stardist(self):
        from csbdeep.utils import normalize
        import numpy as np
        import cv2

        if len(self.eval_image_paths) == 0:
            return None, "No Images"
        
        predictions = []
        self.eval_images = []

        for i, path in enumerate(self.eval_image_paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.eval_images.append(img)

            from backend.data.image_utils import cv2_images_preprocess

            preprocessed = cv2_images_preprocess([img], self.preprocess_settings)[0]
            
            img_resized = cv2.resize(preprocessed, self.model.input_size)
            
            img_norm = normalize(img_resized, 1, 99.8, axis=(0,1)) 
    
            labels, details = self.model.predict_instances(
                img_norm,
                prob_thresh=self.threshold/100,
                nms_thresh=0.3,
                axes='YXC', 
                verbose=False 
            )
            predictions.append(labels)
        
        binary_predictions = []
        for pred in predictions:
            binary = (pred > 0).astype(np.uint8) * 255
            binary_predictions.append(binary)
        
        self.binary_predictions = np.array(binary_predictions)
        self.predictions = self.resize_to_original(binary_predictions)
        self.processed_predictions = self.resize_to_original(predictions)
        
        return predictions, None
    
    def evaluate_cellpose(self):
        from cellpose import models
        import numpy as np
        import cv2

        if len(self.eval_image_paths) == 0:
            return None, "No Images"
        
        predictions = []
        self.eval_images = []

        for i, path in enumerate(self.eval_image_paths):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.eval_images.append(img)

            from backend.data.image_utils import cv2_images_preprocess

            preprocessed = cv2_images_preprocess([img], self.preprocess_settings)[0]
            
            img_resized = cv2.resize(preprocessed, self.model.input_size)
            
            if img_resized.max() <= 1.0:
                img_resized = (img_resized * 255).astype(np.uint8)
        
            channels = [0, 0]  
            diameter = None 
            
            try:
                result = self.model.eval([img_resized], 
                                        diameter=diameter,
                                        channels=channels,
                                        flow_threshold=self.threshold/100,
                                        cellprob_threshold=0.0,
                                        do_3D=False)
                
                if isinstance(result, (tuple, list)):
                    masks = result[0]
                else:
                    masks = result
                
                predictions.append(masks[0])  
                
            except Exception as e:
                print(f"Error using Cellpose: {str(e)}")
                return None, f"Cellpose error: {str(e)}"
        
        binary_predictions = []
        for pred in predictions:
            binary = (pred > 0).astype(np.uint8) * 255
            binary_predictions.append(binary)
        
        self.binary_predictions = np.array(binary_predictions)
        self.predictions = self.resize_to_original(binary_predictions)
        self.processed_predictions = self.resize_to_original(predictions)
        
        return predictions, None

    def evaluate(self):
        import torch
        self.model_name = ""
        
        # Model Selection
        if self.predict_method == "SELECTED_MODEL":
            self.model = self.load_pretrained_model(self.model_selected)
            
            if self.model_name == "stardist":
                return self.evaluate_stardist()
            elif self.model_name == "cellpose":
                return self.evaluate_cellpose()
        
        # Uploaded Model
        elif self.predict_method == "UPLOADED_MODEL":
            self.uploaded_model_init()
            self.image_width = self.model.input_size[0]
            self.image_height = self.model.input_size[1]
            
        # Uploaded Weights
        elif self.predict_method == "UPLOADED_WEIGHTS":
            print(f"{self.weights_uploaded_model_settings.model_type} {self.weights_uploaded_model_settings.model_backbone} {self.weights_uploaded_model_settings.model_framework}")

            if self.weights_uploaded_path == "":
                return None, "No path"
                
            if self.weights_uploaded_model_settings.model_framework == PYTORCH:
                from backend.models.pytorch import PyTorchModel
                self.weights_uploaded = PyTorchModel(self.weights_uploaded_model_settings.model_type,
                                                    self.weights_uploaded_model_settings.model_backbone,
                                                    input_size=(self.image_height, self.image_width))
                self.weights_uploaded.load_weights(path=self.weights_uploaded_path)
                self.framework = PYTORCH
            elif self.weights_uploaded_model_settings.model_framework == KERAS:
                from backend.models.keras import KerasModel
                self.weights_uploaded = KerasModel(self.weights_uploaded_model_settings.model_backbone,
                                                 input_size=(self.image_height, self.image_width, 3))
                self.framework = KERAS
            self.model = self.weights_uploaded

        # Validate model and images
        if self.model is None:
            return None, "No model"
        if len(self.eval_image_paths) == 0:
            return None, "No Images"

        # Process images
        from backend.data.image_utils import paths_to_cv2_images, cv2_images_preprocess, cv2_images_resize, backbone_preprocess

        self.eval_images = paths_to_cv2_images(self.eval_image_paths)
        self.eval_images_preprocessed = cv2_images_preprocess(self.eval_images, self.preprocess_settings)
        self.eval_images_preprocessed = cv2_images_resize(self.eval_images_preprocessed, self.image_width, self.image_height)
        self.eval_images_preprocessed_np = np.array(self.eval_images_preprocessed)
        self.eval_images_preprocessed_np = backbone_preprocess(self.eval_images_preprocessed_np, framework=self.framework, backbone=self.model.backbone)
        
        # Run prediction based on framework
        if self.framework == PYTORCH:
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
        
        else:
            predictions = self.model.predict(self.eval_images_preprocessed_np)
            
            if predictions.ndim == 4:
                self.probabilities = predictions.squeeze(-1)
            else:
                self.probabilities = predictions.copy()
        
        self.threshold_change(self.threshold)
        return self.predictions, None

    def resize_to_original(self, segmented_images):
        resized_images = []
        for i, seg_img in enumerate(segmented_images):
            original_height, original_width = self.eval_images[i].shape[:2] 
            resized_img = cv2.resize(seg_img, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            resized_images.append(resized_img)
        return resized_images

    def threshold_change(self, value):
        self.threshold = value

        if self.predict_method == "SELECTED_MODEL":
            return None

        if len(self.probabilities) > 0:
            if self.framework == PYTORCH: 
                binary_predictions = (self.probabilities > (value/100)).squeeze(1).cpu().numpy()
            else:  
                binary_predictions = (self.probabilities > (value/100))
            
            binary_predictions = (binary_predictions * 255).astype(np.uint8)
            
            self.binary_predictions = binary_predictions
            self.predictions = self.resize_to_original(binary_predictions)
            self.postprocess_cells(binary_predictions)
            return self.processed_predictions[self.preview_image_index]
        return None

    def separate_cells(self, predictions):
        from backend.data.image_utils import (
            apply_standard_watershed, apply_marker_based_watershed,
            apply_distance_transform_watershed, apply_h_minima_watershed,
            apply_compact_watershed
        )
        
        processed_predictions = []
        
        for i in range(len(predictions)):
            binary = predictions[i].astype(np.uint8)
            
            watershed_algorithms = {
                WatershedAlgorithm.STANDARD: apply_standard_watershed,
                WatershedAlgorithm.MARKER_BASED: apply_marker_based_watershed,
                WatershedAlgorithm.DISTANCE_TRANSFORM: apply_distance_transform_watershed,
                WatershedAlgorithm.H_MINIMA: apply_h_minima_watershed,
                WatershedAlgorithm.COMPACT: apply_compact_watershed
            }
            
            algorithm = watershed_algorithms.get(self.current_watershed_algorithm, apply_standard_watershed)
            output = algorithm(binary)
            
            if np.max(output) == 0:
                print(f"No cells detected in image {i}, using original mask")
                output = binary
            
            processed_predictions.append(output)
        
        processed_predictions = np.array(processed_predictions)
        self.processed_predictions = self.resize_to_original(processed_predictions)
  
    def postprocess_cells(self, predictions):
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        try:
            cleaned_predictions = []
            
            for pred in predictions:
                binary = pred.astype(np.uint8)
                cleaned_predictions.append(binary)
            
            cleaned_predictions = np.array(cleaned_predictions)
            
            self.separate_cells(cleaned_predictions)
            
        except Exception as e:
            print(f"Error in postprocess_cells: {str(e)}")
            self.processed_predictions = predictions

    def label_segments(self, current_mask):
        if current_mask.dtype != np.uint8:
            current_mask = current_mask.astype(np.uint8)
            
        colored_mask = cv2.applyColorMap(current_mask, self.current_color_option)
        label_overlay = np.zeros_like(colored_mask)

        unique_labels = np.unique(current_mask)
        unique_labels = unique_labels[unique_labels > 0]  

        metrics_data = []
        id = 1
        for label_id in unique_labels:
            cell_mask = (current_mask == label_id).astype(np.uint8)

            contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)

                circularity = 0.0
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                M = cv2.moments(largest_contour)

                if M["m00"] > 0:  
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])

                    metrics_data.append([id, area, round(circularity, 3)])

                    cv2.putText(label_overlay, str(id), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8, cv2.LINE_AA)
                    id += 1

        result = cv2.addWeighted(colored_mask, 0.8, label_overlay, 1.0, 0)
        return result, metrics_data
    
    def update_watershed_algorithm(self, algo):
        self.current_watershed_algorithm = algo
        
        pretrained_segmentation_models = ["stardist", "cellpose"]
        if self.model_name in pretrained_segmentation_models:
            return None, "Watershed does not work on pretrained segmentation models!", []

        if len(self.binary_predictions) > 0:
            self.postprocess_cells(self.binary_predictions)
            current_mask = self.processed_predictions[self.preview_image_index]
            labeled_image, metrics_data = self.label_segments(current_mask)
            return labeled_image, "", metrics_data
        return None, []