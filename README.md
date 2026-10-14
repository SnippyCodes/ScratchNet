---
title: ScratchNet Flower Classifier
emoji: 🪷
colorFrom: green
colorTo: pink
sdk: gradio
app_file: gradio_app.py
pinned: false
---

# 🪷 ScratchNet — 7-Class Flower Image Classifier & Web App

An end-to-end Deep Learning computer vision project that classifies flower images across **7 distinct botanical categories** using a custom PyTorch Convolutional Neural Network (CNN). Includes an interactive, modern web application (FastAPI), a command-line interface (CLI), a Gradio interface, and a REST API.

---

## 🌺 Detected Flower Categories
The trained model (`flower_classifier.pth`) detects the following 7 flower species:
1. **Daisy** (*Bellis perennis*) — 🌼
2. **Dandelion** (*Taraxacum officinale*) — 🦁
3. **Rose** (*Rosa*) — 🌹
4. **Sunflower** (*Helianthus annuus*) — 🌻
5. **Tulip** (*Tulipa*) — 🌷
6. **Bellflower** (*Campanula*) — 🔔
7. **Lotus** (*Nelumbo nucifera*) — 🪷

---

## 🚀 How to Run & Use the Model

### 1. Interactive Web Application (Recommended)
Launch the modern web UI with drag-and-drop file upload, live image preview, verified 1-click flower samples, real-time confidence scores, and animated probability spectrums:

```bash
# Start the FastAPI server (auto-detects port and opens browser)
py app.py

# Or on local network / Wi-Fi to test from your phone:
py app.py --host 0.0.0.0 --port 8000
```
> **Windows Quick Launch**: You can also just double-click `run_app.bat`!

### 2. Command-Line Interface (CLI)
Classify any image directly from your terminal:

```bash
py predict.py "test_images/download (3).jpg"
# Output:
# 🔔 Top Prediction: BELLFLOWER (100.0%)
# Botanical Name: Campanula
# Hardware:       CUDA
```

### 3. Programmatic Python API
Import the model into your own Python code or Jupyter notebook:

```python
from model import predict_flower

# Run inference on any image path or PIL image
result = predict_flower("my_flower.jpg")

print(f"Prediction: {result['top_flower']} ({result['confidence_pct']})")
print("Top Probabilities:", result['probabilities'])
```

### 4. REST API Endpoint
Send image files via HTTP POST from any client (iOS, Android, React, cURL):

```bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -F "file=@test_images/download.jpg"
```

**Response Format**:
```json
{
  "success": true,
  "top_flower": "lotus",
  "display_name": "Lotus",
  "confidence": 0.605,
  "confidence_pct": "60.5%",
  "ranked_list": [
    {
      "name": "lotus",
      "display_name": "Lotus",
      "probability": 0.605,
      "percentage": "60.5%",
      "color": "#06b6d4",
      "icon": "🪷"
    },
    { "name": "tulip", "percentage": "36.4%", ... }
  ],
  "flower_info": {
    "scientific_name": "Nelumbo nucifera",
    "family": "Nelumbonaceae",
    "icon": "🪷",
    "season": "Summer"
  },
  "device": "cuda"
}
```

### 5. Hugging Face Spaces (Free Global Cloud Deployment)
Run the included Gradio interface or upload to Hugging Face:

```bash
py gradio_app.py
```
To deploy to the world for free:
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new Space (choose **Gradio** SDK).
3. Upload `gradio_app.py`, `model.py`, and `flower_classifier.pth`.
4. Anyone on the internet can now upload photos and test your model!

---

## 🏗️ Neural Network Architecture

```text
Input Image (150 × 150 × 3 RGB) [Normalized / 255.0]
  │
  ├── [Conv Block 1]: Conv2d(3 -> 32, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2)  -> (32, 75, 75)
  ├── [Conv Block 2]: Conv2d(32 -> 64, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2) -> (64, 37, 37)
  ├── [Conv Block 3]: Conv2d(64 -> 128, 3x3) + BatchNorm2d + ReLU + MaxPool2d(2x2)-> (128, 18, 18)
  │
  ├── Flatten -> 41,472 features
  ├── Linear(41,472 -> 128) + ReLU + Dropout(0.3)
  ├── Linear(128 -> 64) + ReLU
  ├── Linear(64 -> 32) + ReLU
  └── Output Linear(32 -> 7) -> Softmax Probabilities (7 Classes)
```

- **Total Parameters**: 5,412,807
- **Test Accuracy**: ~79% on 2,240 test samples
- **Training Epochs**: 20 epochs with Adam (`lr=0.001`) & CrossEntropyLoss
- **Hardware Acceleration**: CUDA RTX 5050 GPU (with automatic CPU fallback)

---

## 🔬 Why Switch from an MLP to a CNN? (Engineering Post-Mortem)

After training our two-layer NumPy Multilayer Perceptron (MLP) for 7,000 iterations and achieving ~90% training accuracy, empirical evaluation on unseen test images exposed a severe generalization bottleneck: real-world accuracy dropped to **40% (4/10)**.

### 1. The Core Limitation: Spatial Oblivion
* **Flattening Destroys Geometry**: An MLP flattens a $150 \times 150 \times 3$ RGB image into a flat vector of $67,500$ independent numbers. The network has no geometric awareness that pixel $(x, y)$ is physically adjacent to pixel $(x+1, y)$.
* **Shortcut Memorization**: The model relied on global color histograms and background pixels rather than flower anatomy. For instance, a dandelion in front of a blue sky was classified as a **Bellflower with 99.0% confidence** because the model associated high blue pixel values at the top coordinates with bellflower backgrounds.
* **Lack of Translation Invariance**: If a flower shifts 10 pixels to the right, every input coordinate shifts to entirely different weight connections in an MLP.

### 2. The Architectural Solution: Convolutional Neural Networks (CNNs)
A CNN decouples the task into two sequential stages:
1. **Feature Extraction (The Visual Cortex)**:
   * **Convolutional Layers (`Conv2d`)**: Slide small $3 \times 3$ parameter-sharing kernels across the spatial grid to detect local features (edges, curves, textures) regardless of where they appear in the frame.
   * **Max Pooling (`MaxPool2d`)**: Progressively downsamples spatial dimensions ($150 \to 75 \to 37 \to 18$), shedding redundant pixel noise, preserving only the strongest feature activations, and enforcing translation invariance.
2. **Classification (The Decision Maker)**:
   * Only after the image is compressed into high-level, meaningful feature representations (e.g., "radiating petal patterns") does a dense linear layer classify the flower.

### 3. Transitioning to PyTorch
While implementing an MLP in pure NumPy provided deep mastery of forward/backward propagation and gradient descent mechanics, building a CNN from scratch in NumPy is computationally impractical on CPU. PyTorch provides:
* **CUDA Hardware Acceleration**: Offloading 2D tensor operations to the RTX 5050 GPU (8 GB VRAM).
* **Dynamic Autograd**: Eliminating manual chain-rule calculus for complex layers like 2D convolutions and Batch Normalization.
