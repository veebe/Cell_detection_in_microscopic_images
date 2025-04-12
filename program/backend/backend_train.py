from PyQt5.QtWidgets import QTableWidgetItem, QFileDialog
from PyQt5.QtCore import QTimer
from frontend.widgets.popUpWidget import PopUpWidget
from backend.training.trainingProgressCallback import TrainingProgressCallback
from backend.training_model import TrainingModel

class TrainingController:
    def __init__(self, ui):
        self.ui = ui
        self.model = TrainingModel()
        
        # Navigation timers
        self.left_timer = QTimer()
        self.right_timer = QTimer()
        self.left_timer.timeout.connect(self.navigate_left)
        self.right_timer.timeout.connect(self.navigate_right)
        
        # Preview index
        self.current_index = 0
        self.test_set_visible = True

    # Image loading methods
    def load_images(self, files):
        count = self.model.load_images(files)
        self.ui.training_tab.image_label.setText(f"Images uploaded: {count}")
        if not self.ui.training_tab.test_set_visible:
            self.ui.training_tab.image_slider.slider.setRange(0, count-1)
            self.ui.training_tab.image_slider.slider.setValue(count // 2)

    def load_masks(self, files):
        count = self.model.load_masks(files)
        self.ui.training_tab.mask_label.setText(f"Masks uploaded: {count}")

    def load_test_images(self, files):
        count = self.model.load_test_images(files)
        self.ui.training_tab.test_image_label.setText(f"Images uploaded: {count}")
        self.ui.training_tab.image_slider.slider.setRange(0, count-1)
        self.ui.training_tab.image_slider.slider.setValue(count // 2)
    
    def load_test_masks(self, files):
        count = self.model.load_test_masks(files)
        self.ui.training_tab.test_mask_label.setText(f"Masks uploaded: {count}")
    
    # Model training
    def train_networks(self):
        # Validate data
        is_valid, error_msg = self.model.validate_train_data()
        if not is_valid:
            popup = PopUpWidget("error", error_msg)
            popup.show()
            return
        
        # Update UI
        self.ui.training_tab.train_button.setText("Training...") 
        self.ui.training_tab.train_button.setEnabled(False)
        self.ui.training_tab.progress_bar.setMaximum(self.model.model_settings.epochs)
        self.ui.training_tab.metrics_table.clear_table()

        # Get image size from UI
        image_size_str = self.ui.training_tab.image_size_dropdown.currentText().split('x')
        image_size = (int(image_size_str[0]), int(image_size_str[0]))
        
        # Prepare data
        X_train, X_val, y_train, y_val = self.model.prepare_training_data(image_size)
        
        # Setup callback
        self.callback = TrainingProgressCallback(self.model.model_settings.epochs)
        self.callback.progress_updated.connect(self.update_process_bar)
        self.callback.training_completed.connect(self.on_training_finished)
        self.callback.training_failed.connect(self.on_training_error)
        self.callback.metrics_updated.connect(self.update_metrics_table)
        
        # Update slider if not using dedicated test set
        if not self.ui.training_tab.test_set_visible:
            max_index = len(self.model.val_indices) - 1
            self.ui.training_tab.image_slider.slider.setRange(0, max_index)
            self.current_index = int(max_index / 2)
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)
        
        # Create model
        self.model.create_model()
        
        # Train based on framework
        if self.model.model_settings.model_framework == "keras":
            train_generator = self.model.setup_keras_training(X_train, y_train)
            self.model.model._train(train_generator, (X_val, y_val), callbacks=self.callback)
        else:  # PyTorch
            train_loader, val_loader = self.model.setup_pytorch_training(X_train, y_train, X_val, y_val)
            self.model.model._train(train_loader, val_loader, callbacks=self.callback)

    def on_training_finished(self):
        # Generate predictions on validation set
        self.model.predict_on_validation()
        
        # Display results
        self.display_current_image()
        
        # Update UI
        self.ui.training_tab.download_model_button.setEnabled(True)
        self.ui.training_tab.download_weights_button.setEnabled(True)
        self.ui.training_tab.train_button.setEnabled(True)
        self.ui.training_tab.train_button.setText("Train and detect")
        self.ui.training_tab.progress_bar.setValue(0)

    def on_training_error(self, error_msg):
        print(f"Error: {error_msg}")
        self.ui.training_tab.train_button.setText("Train and Detect") 
        self.ui.training_tab.train_button.setEnabled(True)
        popup = PopUpWidget("error", error_msg)
        popup.show()

    def update_metrics_table(self, metrics):
        row_position = self.ui.training_tab.metrics_table.table.rowCount()
        self.ui.training_tab.metrics_table.table.insertRow(row_position)
        
        self.ui.training_tab.metrics_table.table.setItem(row_position, 0, QTableWidgetItem(str(metrics['epoch'])))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 1, QTableWidgetItem(metrics['loss']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 2, QTableWidgetItem(metrics['accuracy']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 3, QTableWidgetItem(metrics['val_loss']))
        self.ui.training_tab.metrics_table.table.setItem(row_position, 4, QTableWidgetItem(metrics['val_accuracy']))
        
        self.ui.training_tab.metrics_table.table.scrollToBottom()

    def update_process_bar(self, value):
        print(value)
        self.ui.training_tab.progress_bar.setValue(value)

    def display_current_image(self):
        # Get image and mask paths
        image_path, mask_path = self.model.get_original_image_and_mask(self.current_index)
        
        if not self.ui.training_tab.test_set_visible:
            if image_path:
                self.ui.training_tab.image_drop.display_image(image_path)
            if mask_path:
                self.ui.training_tab.mask_drop.display_image(mask_path)
        else:
            if image_path:
                self.ui.training_tab.test_image_drop.display_image(image_path)
            if mask_path:
                self.ui.training_tab.test_mask_drop.display_image(mask_path)

        # Get predicted mask
        predicted_mask = self.model.get_validation_image(self.current_index)
        if predicted_mask is not None:
            display_mask = predicted_mask.squeeze()
            self.ui.training_tab.plot.ax.imshow(display_mask, cmap='gray')
            self.ui.training_tab.plot.ax.set_title("Predicted Mask")
        else:
            print("Error: No predicted mask available.")
        
        self.ui.training_tab.plot.canvas.draw()
    
    # Navigation methods
    def start_navigate_left(self):
        self.navigate_left()  
        self.left_timer.start(100)  

    def start_navigate_right(self):
        self.navigate_right()  
        self.right_timer.start(100) 

    def stop_navigate(self):
        self.left_timer.stop()
        self.right_timer.stop()

    def navigate_left(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)

    def navigate_right(self):
        image_count = len(self.model.processed_images) - 1
        if image_count > 0 and self.current_index < image_count:
            self.current_index += 1
            self.display_current_image()
            self.ui.training_tab.image_slider.slider.setValue(self.current_index)
    
    def move_preview(self, value):
        self.current_index = value
        self.display_current_image()
    
    # Toggle test set visibility
    def toggle_training_set(self):
        self.test_set_visible = not self.test_set_visible
        self.model.using_dedicated_testing_set = self.test_set_visible
        
        if not self.test_set_visible:
            self.ui.training_tab.image_slider.slider.setRange(0, len(self.model.image_paths) - 1)
            self.ui.training_tab.image_slider.slider.setValue(len(self.model.image_paths) // 2)
        else:
            self.ui.training_tab.image_slider.slider.setRange(0, len(self.model.test_image_paths) - 1)
            self.ui.training_tab.image_slider.slider.setValue(len(self.model.test_image_paths) // 2)
    
    # Save settings
    def save_settings(self, values):
        self.model.model_settings.save_settings(values=values)
        self.model.preprocess_settings.save_settings(values=values)
        print("Model Settings Updated:", self.model.model_settings.__dict__)
        print("Preprocessing Settings Updated:", self.model.preprocess_settings.__dict__)

    # Save model/weights
    def download_model(self):
        if not hasattr(self.model, 'model') or self.model.model is None:
            popup = PopUpWidget("error", "No model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model.model_settings.model_framework == "keras":
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.keras", "Keras Files (*.keras);;H5 Files (*.h5);;All Files (*)", options=options)
        else:  # PyTorch
            file_path, _ = QFileDialog.getSaveFileName(None, "Save model", "model.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            success = self.model.save_model(file_path)
            if not success:
                popup = PopUpWidget("error", "Failed to save model!")
                popup.show()

    def download_weights(self):
        if not hasattr(self.model, 'model') or self.model.model is None:
            popup = PopUpWidget("error", "No model found!")
            popup.show()
            return

        options = QFileDialog.Options()
        if self.model.model_settings.model_framework == "keras":
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.weights.h5", "H5 Files (*.h5);;All Files (*)", options=options)
        else:  # PyTorch
            file_path, _ = QFileDialog.getSaveFileName(None, "Save weights", "weights.pth", "Pth Files (*.pth);;All Files (*)", options=options)

        if file_path:
            success = self.model.save_weights(file_path)
            if not success:
                popup = PopUpWidget("error", "Failed to save weights!")
                popup.show()