# **Realistic ECG to Time Series Converter** 🫀📈

This repository contains an end-to-end AI pipeline that takes realistic, un-normalized ECG images, crops the individual leads, converts them to clean black-and-white signals, and finally extracts the 1D time-series data. 

## 🚀 Pipeline Overview

The project is divided into 4 sequential stages, taking raw clinical-style ECGs and systematically transforming them into actionable data:

1. **Data Generation (`src/final_image_generator.ps1`)**: A PowerShell orchestration script that interfaces with an ECG generator kit to build raw training batches in standard and B&W formats.
2. **Lead Cropping (`src/Crop_Code.txt`)**: Utilizes an Object Detection model (Detectron2 / Faster R-CNN) to detect bounding boxes for standard ECG leads (I, II, III, aVR, aVL, aVF, V1-V6, rhythm) and crop them automatically.
3. **Real to B&W Masking (`src/Real_To_B&W_Code.txt`)**: A deep learning segmentation model (using `segmentation_models_pytorch` with Unet/ResNet) that isolates the ECG signal wave from gridlines and background noise, outputting a clean B&W mask.
4. **Time Series Extraction (`src/BW_TS_Un_Normalized_Code.txt`)**: A PyTorch-based custom model that translates the 2D B&W signal crop into a 1D time-series CSV format, optimized using Cosine Annealing and Huber Loss.

## 📁 Repository Structure

ECG-to-TimeSeries/
│
├── data/                      # Sample inputs/outputs here
├── src/                       # Pipeline Source Code
│   ├── final_image_generator.ps1
│   ├── Crop_Code.txt          # (Python code for Detectron2)
│   ├── Real_To_B&W_Code.txt   # (Python code for Segmentation)
│   └── BW_TS_Un_Normalized_Code.txt # (Python code for Time Series)
│
├── inference_images/          # Sample outputs and loss graphs
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/ECG-to-TimeSeries.git](https://github.com/yourusername/ECG-to-TimeSeries.git)
   cd ECG-to-TimeSeries
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Detectron2 (Required for Cropping):**
   Detectron2 requires a custom installation depending on your PyTorch and CUDA versions. 
   ```bash
   pip install 'git+[https://github.com/facebookresearch/detectron2.git](https://github.com/facebookresearch/detectron2.git)'
   ```
   *For specific system requirements, refer to the [official Detectron2 installation guide](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).*

## 🧠 Models Used

* **Object Detection (Cropping):** Detectron2 Faster R-CNN (`COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml`)
* **Segmentation (B&W Masking):** PyTorch Segmentation Models (SMP) with Albumentations for augmentation.
* **Time Series Generation:** Custom PyTorch architecture utilizing `ReduceLROnPlateau` and `CosineAnnealingLR` schedulers.
