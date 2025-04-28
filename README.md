# 🔬 Cell Detection App

## 📄 Overview

This application detects and segments cells in microscopic images using deep learning models. It provides options for image preprocessing, segmentation post-processing, and supports various segmentation architectures.

## ✨ Features

* **Deep Learning-Based Segmentation**: Utilizes models like U-Net, U-Net++, and others for precise cell detection and segmentation.
* **Custom Preprocessing**: Apply enhancements such as denoising and blurring before training or segmentation.
* **Interactive Preview**: Visualize segmentation results dynamically.
* **Watershed Algorithm**: Includes additional segmentation algorithms like Watershed with customizable thresholding options.

## 🚀 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/veebe/Cell_detection_in_microscopic_images.git
    cd Cell_detection_in_microscopic_images
    ```

2.  **Create and Activate a Virtual Environment**
    * Create the environment:
        ```bash
        python -m venv venv
        ```
    * Activate it:
        * Windows (cmd/powershell):
            ```bash
            .\venv\Scripts\activate
            ```
3.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

4.  **⚠️ Install GPU Support (CUDA)**

    * **NVIDIA CUDA Toolkit**: This application relies heavily on GPU acceleration via CUDA. You **must** install the NVIDIA CUDA Toolkit suitable for your GPU and driver version. Download it from the [official NVIDIA site](https://developer.nvidia.com/cuda-downloads).
    * **PyTorch with CUDA**: You also need to install the PyTorch version that matches your **specific installed CUDA Toolkit version**.
        * Go to the [official PyTorch website](https://pytorch.org/get-started/locally/).
        * Use their configuration tool to select your OS, `Pip`, Python, and importantly, the **CUDA version** you installed (e.g., CUDA 12.6, CUDA 11.8).
        * Copy and run the generated `pip install` command. It will look something like this ( **do not copy this example directly, use the one generated for *your* system!** ):
            ```bash
            # Example for CUDA 12.1 - Use the command from the PyTorch website!
            pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
            ```
    * 🚨 **Platform Disclaimer**: This application has been primarily developed and tested on systems with NVIDIA GPUs using CUDA. Functionality on other platforms (CPU-only, AMD GPUs) is not guaranteed and has not been tested.

5.  **Run the Application**
    If all installations completed successfully, you can launch the main application script:
    ```bash
    python .\program\main.py
    ```

## ▶️ Usage

### 🎓 Training

1.  Load your microscopic images and their corresponding segmentation masks.
2.  Choose desired preprocessing options (e.g., bluring, contrast adjustment).
3.  Configure the model options (framework, specific model architecture, backbone, number of epochs, etc.).
4.  Set the training dimensions (e.g., input image size).
5.  Start the training process.
6.  View training progress and save the trained model or weights upon completion.

### 🎨 Segmentation

1.  Load the microscopic images you want to segment.
2.  Select the detection method.
3.  Load your trained model/weights, or choose one of the available pre-trained models.
4.  Apply any desired preprocessing steps (e.g., blurring, contrast adjustment).
5.  Run the segmentation process.
6.  Interactively adjust parameters like thresholding or Watershed settings to refine the segmentation results.
7.  View and save the final segmentation metrics.

## 📜 License

Closed Source

## 📧 Contact

For questions, please contact: vratislav.blunar.st@vsb.cz
