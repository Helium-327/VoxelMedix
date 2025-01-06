# VoxelMedix —— 🧠 BraTS 2021 3D MRI Segmentation Project 🧠

> 欢迎来到 BraTS 2021 3D 核磁共振成像分割项目！本项目侧重于利用深度学习技术从三维核磁共振扫描中分割脑肿瘤。目标是准确识别和分类不同的肿瘤区域，如增强肿瘤（ET）、肿瘤核心（TC）和整个肿瘤（WT）。

## 📂 Project Structure

```txt
📦 src
├── 📄 BraTS.py               # Dataset loading and preprocessing
├── 📄 inference.py           # Inference and prediction logic
├── 📄 loss_function.py       # Custom loss functions (Dice, Focal, CE)
├── 📄 main.py                # Main script for training and inference
├── 📄 metrics.py             # Metrics and evaluation utilities
├── 📄 train_and_val.py       # Training and validation logic
├── 📄 train.py               # Training loop and model management
```


## 🛠️ Dependencies

- **PyTorch**: For deep learning model training and inference.
- **Nibabel**: For handling NIfTI files (MRI scans).
- **Numpy**: For numerical operations.
- **Pandas**: For data manipulation.
- **TorchIO**: For medical image augmentation and preprocessing.

To set up the environment, you can use the following commands:

```python
# Using conda
conda env create -f environment.yml
conda activate cv

# Using pip
pip install -r requirements.txt
```


## 🚀 Getting Started

### Data Preparation

### Training the Model

### Inference



## 🧩 Key Features


## 📊 Metrics


## 📝 Notes


## 🤝 Contributing



## 📜 License
