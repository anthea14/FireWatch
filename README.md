# 🔥 FireWatch — Satellite Wildfire Detection System

A deep learning-based wildfire detection system that classifies satellite images as **Wildfire** or **No Wildfire** using a fine-tuned ConvNeXt-Tiny model. Includes Grad-CAM explainability, calibrated confidence scores, and a FastAPI web application.

---

## Demo

Upload any satellite image through the web UI and get:
- Wildfire / No Wildfire prediction
- Calibrated confidence score with risk tier (Low / Elevated / Critical)
- Grad-CAM heatmap highlighting the fire regions in the image

---

## Model Benchmark Results

All models sorted by Recall (most critical metric — missing a real fire is far costlier than a false alarm):

| Model | Accuracy | F1 | Precision | Recall | AUC-ROC | Inference (ms) | Size (MB) |
|---|---|---|---|---|---|---|---|
| **convnext_tiny** ⭐ | 99.62% | 99.66% | 99.63% | **99.68%** | 99.92% | 6.37 | 111.36 |
| tf_efficientnetv2_s | 99.32% | 99.38% | 99.48% | 99.28% | 99.96% | 18.85 | 81.63 |
| resnet50 | 98.95% | 99.05% | 99.11% | 98.99% | 99.92% | 5.62 | 94.37 |
| mobilenetv3_large_100 | 99.03% | 99.12% | 99.36% | 98.88% | 99.89% | 6.09 | 17.03 |

🏆 **Best model by recall: convnext_tiny**

---

## Project Structure
wildfire-app/
├── app/
│   └── main.py              ← FastAPI backend
├── models/
│   ├── best_model.pt        ← Download from Kaggle (see below)
│   ├── deploy_config.json
│   └── benchmark_results.json
├── templates/
│   └── index.html           ← Frontend UI
├── requirements.txt
└── Dockerfile

---

## Setup & Running Locally

### 1. Clone the repo

```bash
git clone https://github.com/your-username/firewatch.git
cd firewatch
```

### 2. Download model weights

The model file `best_model.pt` (111MB) is not included in this repo due to GitHub's file size limit.

Download it from the Kaggle notebook outputs and place it at:
wildfire-app/models/best_model.pt

### 3. Install dependencies

For CPU-only (recommended if no GPU):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

For GPU (check your CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/)):
```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
cd wildfire-app
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

---

## How It Works

1. Satellite image is uploaded through the web UI
2. Image is resized to 224×224 and normalized with ImageNet stats
3. ConvNeXt-Tiny backbone runs inference and outputs a calibrated confidence score
4. Risk tier is assigned — 🟢 Low (0–29) / 🟡 Elevated (30–59) / 🔴 Critical (60–100)
5. Grad-CAM generates a heatmap showing which regions influenced the prediction

---

## Training

The model was trained on the [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) on Kaggle using a free T4 GPU.

**Fine-tuning strategy:**
- Phase 1 (5 epochs): Freeze backbone, train classification head only
- Phase 2 (10 epochs): Unfreeze all layers, cosine LR decay

**4 architectures were benchmarked:** ResNet-50, EfficientNetV2-S, MobileNetV3-Large, ConvNeXt-Tiny

ConvNeXt-Tiny was selected as the best model based on highest recall.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Training | PyTorch, timm, torchmetrics, pytorch-grad-cam |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Explainability | Grad-CAM |
| Deployment | Docker, Hugging Face Spaces |

---

## Deployment (Hugging Face Spaces)

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces) → select **Docker**
2. Push this repo as a Git repository
3. Upload `best_model.pt` via the HF web UI or Git LFS
4. HF will build and serve automatically on port 7860

---

## Results

- 99.68% Recall — misses almost no real wildfire
- 99.62% Accuracy across the full test set
- 99.92% AUC-ROC — near-perfect class separation
- Grad-CAM heatmaps consistently highlight fire regions and smoke plumes
- Fully runs on CPU — no GPU required at inference

---

## Acknowledgements

- Dataset: [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset) by abdelghaniaaba on Kaggle
- Backbone models via [timm](https://github.com/huggingface/pytorch-image-models)
- Explainability via [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)
