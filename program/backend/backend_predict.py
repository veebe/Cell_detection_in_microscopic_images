from PyQt5.QtCore import QTimer
from frontend.widgets.popUpWidget import PopUpWidget
from backend.prediction_model import PredictionModel
from backend.backend_types import PredictMethods
from backend.setting_classes import ModelSettings

class PredictionController:
    def __init__(self, ui):
        self.ui = ui
        self.model = PredictionModel()
        self.weights_uploaded_model_settings = ModelSettings()
        
        # Navigation timers
        self.left_timer = QTimer()
        self.right_timer = QTimer()
        self.left_timer.timeout.connect(self.predict_navigate_left)
        self.right_timer.timeout.connect(self.predict_navigate_right)

    # Model loading methods
    def load_model(self, path):
        self.model.load_model(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def load_metadata(self, path):
        self.model.load_metadata(path)
        
    # Set prediction method
    def set_predict_method(self, method):
        if method == "Uploaded model":
            self.model.predict_method = "UPLOADED_MODEL"
        elif method == "Uploaded weights":
            self.model.predict_method = "UPLOADED_WEIGHTS"
        elif method == "Pretrained model":
            self.model.predict_method = "SELECTED_MODEL"
    
    # Set model selection
    def set_model_selected(self, model_name):
        self.model.model_selected = model_name.lower()
    
    # Image loading
    def set_eval_images(self, paths):
        self.model.eval_image_paths = paths
        
    # Evaluation
    def evaluate(self):
        # Transfer settings from controller to model
        self.model.weights_uploaded_model_settings = self.weights_uploaded_model_settings
        
        # Get image dimensions from UI if using uploaded weights
        if self.model.predict_method == "UPLOADED_WEIGHTS":
            self.model.image_height = int(self.ui.analysis_tab.image_size_dropdown.currentText().split('x')[0]) 
            self.model.image_width = int(self.ui.analysis_tab.image_size_dropdown.currentText().split('x')[0])
            
        # Run evaluation
        predictions, error = self.model.evaluate()
        
        # Display error if any
        if error:
            popup = PopUpWidget("error", error)
            popup.show()
            return None
            
        # Display results
        self.disply_current_predicted_image()
        return predictions
    
    # Update display
    def disply_current_predicted_image(self):
        self.ui.analysis_tab.eval_images_drop.display_image(self.model.eval_image_paths[self.model.preview_image_index])
        if len(self.model.predictions) > 0:
            self.ui.analysis_tab.predicted_image.display_image(self.model.predictions[self.model.preview_image_index])

            current_mask = self.model.processed_predictions[self.model.preview_image_index]
            labeled_image, metrics_data = self.model.label_segments(current_mask)
            self.ui.analysis_tab.segmented_image.display_image(labeled_image)
            
            # Update metrics table
            self.ui.analysis_tab.metrics_table.clear_table()
            for row_data in metrics_data:
                self.ui.analysis_tab.metrics_table.add_row(row_data)
    
    # Color option update
    def update_color_option(self, color):
        self.model.current_color_option = color
        self.disply_current_predicted_image()
    
    # Watershed algorithm update
    def update_watershed_algorithm(self, algo):
        labeled_image, error_msg, metrics_data = self.model.update_watershed_algorithm(algo)
    
        if error_msg:
            popup = PopUpWidget("warning", error_msg)
            popup.show()
            return
                
        if labeled_image is not None:
            self.ui.analysis_tab.segmented_image.display_image(labeled_image)
                
            # Update metrics table
            self.ui.analysis_tab.metrics_table.clear_table()
            for row_data in metrics_data:
                self.ui.analysis_tab.metrics_table.add_row(row_data)
    
    # Threshold change
    def threshold_change(self, value):
        current_mask = self.model.threshold_change(value)
        if current_mask is not None:
            labeled_image, metrics_data = self.model.label_segments(current_mask)
            self.ui.analysis_tab.predicted_image.display_image(self.model.predictions[self.model.preview_image_index])
            self.ui.analysis_tab.segmented_image.display_image(labeled_image)
            
            # Update metrics table
            self.ui.analysis_tab.metrics_table.clear_table()
            for row_data in metrics_data:
                self.ui.analysis_tab.metrics_table.add_row(row_data)
    
    # Navigation methods
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
        if self.model.preview_image_index > 0:
            self.model.preview_image_index -= 1
            self.disply_current_predicted_image()
            self.ui.analysis_tab.image_slider.slider.setValue(self.model.preview_image_index)

    def predict_navigate_right(self):
        if self.model.preview_image_index < len(self.model.eval_image_paths) - 1:
            self.model.preview_image_index += 1
            self.disply_current_predicted_image()
            self.ui.analysis_tab.image_slider.slider.setValue(self.model.preview_image_index)
    
    def predict_move_preview(self, value):
        self.model.preview_image_index = value
        self.disply_current_predicted_image()
    
    # Settings
    def save_settings(self, values):
        self.model.preprocess_settings.save_settings(values=values)
        print("Preprocessing Settings Updated:", self.model.preprocess_settings.__dict__)