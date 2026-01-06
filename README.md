# 🎯 Face Detection System

A face detection system built with YOLOv8, featuring an intuitive GUI and comprehensive evaluation tools.

## 📋 Overview

Complete face detection pipeline using **YOLOv8**, from dataset preprocessing to model training, inference, and evaluation. Features a **Tkinter GUI** with threading support for responsive user experience.

### ✨ Key Features

- ✅ **YOLOv8 Nano** - Fast and accurate face detection
- ✅ **Modern GUI** - Intuitive interface with real-time feedback
- ✅ **Complete Pipeline** - Preprocessing → Training → Detection → Evaluation
- ✅ **Batch Processing** - Process multiple images efficiently
- ✅ **Comprehensive Metrics** - Precision, Recall, F1-Score with IoU matching
- ✅ **Threading Support** - Non-blocking UI during operations

## 📊 Dataset

This project uses the **Human Faces Object Detection Dataset** from Kaggle:

🔗 **Dataset Source:** [Human Faces (Object Detection)](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection)

### Dataset Details

- **Total Images:** 2,204 face images
- **Annotations:** Bounding box coordinates in CSV format
- **Original Format:** `x0, y0, x1, y1` (top-left and bottom-right corners)
- **Use Case:** Face detection and localization

### Train/Test Split

The dataset was split for training and evaluation:

- **Training Set:** 80% (~1,763 images)
- **Test/Validation Set:** 20% (~441 images)
- **Split Method:** Random split with seed for reproducibility

### Preprocessing Pipeline

The system converts Kaggle CSV format to YOLO format:

**Input (Kaggle CSV):**
```csv
image_name,width,height,x0,y0,x1,y1
img1.jpg,800,600,100,150,200,250
```

**Output (YOLO Format):**
```
class_id x_center y_center width height
0 0.2125 0.3583 0.1250 0.2167
```

All coordinates are **normalized to [0, 1]** relative to image dimensions.

### Download Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/sbaghbidi/human-faces-object-detection)
2. Extract to `data/` directory
3. Split into `train/` and `test/` folders
4. Run preprocessing to convert to YOLO format

### Dataset Structure

```
data/
├── train/
│   ├── images/          # Training images
│   └── annotations.csv  # Bounding boxes (x0,y0,x1,y1)
└── test/
    ├── images/          # Test images
    └── annotations.csv  # Ground truth
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/violetahoza/Face-Detection.git
cd face_detection

# Install dependencies
pip install -r requirements.txt
```

### Run Application

```bash
python main.py
```

### Workflow

1. **Preprocess** - Convert CSV to YOLO format
2. **Train** - Fine-tune YOLOv8 on your dataset  
3. **Detect** - Find faces in new images
4. **Evaluate** - Get precision, recall, F1-score

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Training
DEFAULT_EPOCHS = 30
DEFAULT_BATCH_SIZE = 16
DEVICE = 'cpu'  # Change to 'cuda' for GPU

# Detection
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
```

### Detection Pipeline

```
Input → Resize (letterbox) → Normalize → 
YOLOv8 Inference → NMS → Scale back → Output
```

All preprocessing, NMS, and coordinate scaling happen **inside Ultralytics YOLO**.

## 📈 Evaluation

Metrics based on IoU matching (threshold: 0.5):

- **Precision:** % of detections that are correct
- **Recall:** % of faces that were detected  
- **F1-Score:** Harmonic mean of precision & recall

Example output:
```
Precision: 0.9491 (94.91%)
Recall: 0.9111 (91.11%)
F1-Score: 0.9297 (92.97%)
```

## 📁 Project Structure

```
├── config.py           # Central configuration
├── csv_parser.py       # Parse CSV annotations
├── preprocessing.py    # Dataset conversion
├── training.py         # Model training
├── detection.py        # Detection & evaluation
├── main.py            # GUI application
├── requirements.txt   # Dependencies
│
├── data/              # The dataset
├── faces_dataset/     # YOLO format (generated)
├── yolo_training/     # Models & metrics (generated)
└── outputs/           # Detection results (generated)
```

## 🐛 Troubleshooting

**Training is slow?**
- Set `DEVICE = 'cuda'` in `config.py` for GPU
- Reduce batch size if out of memory
- Use fewer epochs for testing

**No faces detected?**
- Lower `DEFAULT_CONF_THRESHOLD`
- Ensure faces are similar to training data
- Check minimum face size (>16px recommended)

**Model not found?**
- Run preprocessing before training
- Check `yolo_training/face_detector/weights/best.pt` exists

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [Tkinter](https://docs.python.org/3/library/tkinter.html)

---

