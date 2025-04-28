# Cell Detection App

## Overview
This application detects and segments cells in microscopic images using deep learning models. It provides preprocessing options, postrocessing segmentation and supports multiple segmentation architectures.

## Features
- **Deep Learning-Based Segmentation**: Uses U-Net, U-Net++ and more for cell detection.
- **Custom Preprocessing**: Apply denoising, blur and other enhancements before training and segmentation.
- **Real-Time Preview**: Click and hold and watch your segmented images move in time.
- **Watershed Algorithm**: Additional algorithms for segmentation, custom tresholding.


## 🚀 Installation

1. **Clone the repository**
   ```
   git clone https://github.com/veebe/Cell_detection_in_microscopic_images.git
   ```
   navigate to the cloned directory and create a virtual enviroment
   ```
   python -m venv venv
   .\venv\Scripts\activate
   ```
   install the requirements using
   ```
   pip install -r requirements.txt
   ```
   ⚠️ WARNING ⚠️ - you also need to have nvidia CUDA toolkit installed for PyTorch and TensorFlow to work properly, download the CUDA toolkit from provided site https://developer.nvidia.com/cuda-downloads
   for PyTorch you need to go to https://pytorch.org and select the installation based on your requirements. On my machine that was
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   🚨 DISCLAIMER 🚨 - the application has been build mainly for the CUDA platform, other platforms may or may not work (they have not been tested!)

   if all installs completed succesfully, you can run the main.py that is located in the /program directory
   ```
   python .\program\main.py
   ```

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
4. Choose preprocessing options (blur, contrast, etc.).
6. Run the detection process.
7. Play with the treshold and watershed methods to get the expected result
8. View the segmented image and masks.

## License
closed source

## Contact
For questions contact vratislav.blunar.st@vsb.cz

