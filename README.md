# Waste Classification with Deep Learning

A binary image classification project that uses transfer learning to automatically sort waste into two categories: **Organic (O)** and **Recyclable (R)**. The model is built on MobileNetV2 and trained using TensorFlow/Keras on Google Colab with GPU acceleration.

---

## Overview

This project applies computer vision to the problem of waste sorting. Given an image of waste, the model predicts whether the item is organic or recyclable — a practical approach that could support automated recycling systems and smart waste management pipelines.

---

## Model Architecture

The model uses MobileNetV2 pretrained on ImageNet as a feature extractor, with a custom classification head added on top:

- GlobalAveragePooling2D
- Dropout (0.3)
- Dense (2 units, softmax activation)

Input size is 160x160x3 and the output is a probability distribution over the two classes.

Training was done in two phases. In the first phase, the base model was frozen and only the custom head was trained for 10 epochs with a learning rate of 0.0005. In the second fine-tuning phase, the top layers of the base model (from layer 100 onward) were unfrozen and trained for 3 additional epochs with a learning rate of 0.00001.

---

## Results

The model reached approximately **93% training accuracy** and **92.8% validation accuracy** by the end of training. On the test set, individual predictions showed confidence scores above 0.95 in most cases, with many samples reaching 1.00 confidence.

---

## Dataset

The dataset is sourced from Kaggle: [Waste Classification Data by techsash](https://www.kaggle.com/datasets/techsash/waste-classification-data).

It contains images split into two classes — O (Organic) and R (Recyclable). The data was further divided as follows:

- Train: ~19,181 images
- Validation: ~3,383 images (15% split from the training set)
- Test: ~2,513 images

---

## Project Structure

```
WASTE-classifcation/
├── archive/
│   └── DeepLearningProject (2).ipynb
└── README.md
```

---

## Getting Started

### Requirements

```
tensorflow
opencv-python
kaggle
numpy
```

Install them with:

```bash
pip install tensorflow opencv-python kaggle numpy
```

### Setting Up Kaggle

Place your Kaggle credentials in `~/.kaggle/kaggle.json`:

```json
{"username": "your_username", "key": "your_api_key"}
```

Then set the correct permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Downloading the Dataset

```bash
kaggle datasets download -d techsash/waste-classification-data -p /content/waste_project --unzip
```

### Running the Notebook

Open the notebook in Google Colab and select a T4 GPU runtime. Run all cells in order. The notebook handles data loading, augmentation, model building, training, fine-tuning, and inference.

---

## Inference

To run a prediction on a single image:

```python
import cv2
import numpy as np

def predict_image(path, model, labels, img_size=(160, 160)):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    x = np.expand_dims(img.astype('float32') / 255.0, 0)
    pred = model.predict(x)[0]
    idx = int(np.argmax(pred))
    return labels[idx], float(pred[idx])

labels = {0: 'O', 1: 'R'}
label, confidence = predict_image("your_image.jpg", model, labels)
print(f"Predicted: {label} | Confidence: {confidence:.2f}")
```

---

## Tech Stack

- Python 3
- TensorFlow / Keras
- MobileNetV2
- OpenCV
- NumPy
- Google Colab (T4 GPU)
- Kaggle API

---

## License

This project is open source and available for educational and research purposes.
