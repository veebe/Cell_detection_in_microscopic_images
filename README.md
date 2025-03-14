# Cell Detection App

## Overview
This application detects and segments cells in microscopic images using deep learning models. It provides preprocessing options, postrocessing segmentation and supports multiple segmentation architectures.

## Features
- **Deep Learning-Based Segmentation**: Uses U-Net, U-Net++ and more for cell detection.
- **Custom Preprocessing**: Apply denoising, blur and other enhancements before training and segmentation.
- **Real-Time Preview**: Click and hold and watch your segmented images move in time.
- **Watershed Algorithm**: Additional algorithms for segmentation, custom tresholding.

## Installation
### Requirements
- Python 3.8+
- PyTorch
- TensorFlow/Keras
- OpenCV
- Qt for GUI
- NumPy, Matplotlib
- more (I will create requirements.txt with everything needed)

## Usage
### Training
1. Load a microscopic images and masks.
2. Choose preprocessing options (denoising, thresholding, etc.).
3. Choose model options (framework, model, backbone, epochs, etc.).
4. Select training dimensions.
6. Run the training process.
7. View and save the model or weights.

### Segmentation
1. Load a microscopic images.
2. Select detection method.
      * Load the selected detection model, weights or select one from the pretrained models.
4. Choose preprocessing options (denoising, thresholding, etc.).
6. Run the detection process.
7. View the segmented image and masks.

## License
closed source

## Contact
For questions contact vratislav.blunar.st@vsb.cz

