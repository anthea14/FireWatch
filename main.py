"""
Wildfire Detection API — FastAPI backend
Run: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Fixes applied (vs original):
  1. Real temperature scaling — T learned on validation set, stored in deploy_config.json
  2. Grad-CAM targets the PREDICTED class, not always FIRE_IDX
  3. "Risk score" renamed to "confidence score" with honest UI framing
  4. Input validation: file-size cap + PIL error handling
  5. Inference time now excludes Grad-CAM (reported separately)
  6. Model load uses map_location=DEVICE unconditionally (GPU→CPU safety)
  7. Grad-CAM toggle (include_gradcam query param, default True)
"""
import io, json, time, base64
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────
ROOT = Path("C:/Users/shann/Downloads/files")

MODELS_DIR   = ROOT / "models"
STATIC_DIR   = ROOT / "static"
TEMPLATE_DIR = ROOT / "templates"

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB hard cap

# ── Load deploy config ────────────────────────────────────────────────────
CFG_PATH = MODELS_DIR / "deploy_config.json"
if not CFG_PATH.exists():
    DEPLOY_CFG = {
        "model_name"  : "resnet50",
        "num_classes" : 2,
        "class_names" : ["nowildfire", "wildfire"],
        "fire_idx"    : 1,
        "img_size"    : 224,
        "temperature" : 1.0,   # no calibration in demo mode
        "metrics"     : {},
    }
    DEMO_MODE = True
else:
    with open(CFG_PATH) as f:
        DEPLOY_CFG = json.load(f)
    DEMO_MODE = False

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = DEPLOY_CFG["img_size"]
FIRE_IDX    = DEPLOY_CFG["fire_idx"]
NO_FIRE_IDX = 1 - FIRE_IDX
CLASS_NAMES = DEPLOY_CFG["class_names"]

# ── Temperature (learned on validation set, saved by notebook) ────────────
# If not present in config, fall back to 1.0 (= no rescaling).
# FIX #1: temperature must come from deploy_config.json, NOT a hardcoded guess.
TEMPERATURE = float(DEPLOY_CFG.get("temperature", 1.0))

# ── Image pre-processing ──────────────────────────────────────────────────
PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
INV_NORM = T.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229,       1/0.224,       1/0.225],
)

# ── Load model ────────────────────────────────────────────────────────────
def load_model() -> Optional[torch.nn.Module]:
    ckpt = MODELS_DIR / "best_model.pt"
    if not ckpt.exists():
        print("⚠  No weights found at models/best_model.pt — running in DEMO MODE")
        return None
    model = timm.create_model(
        DEPLOY_CFG["model_name"],
        pretrained=False,
        num_classes=DEPLOY_CFG["num_classes"],
    )
    # FIX #5: always pass map_location so GPU-trained checkpoints load on CPU
    state = torch.load(ckpt, map_location=DEVICE)
    # Support both raw state-dicts and {'model_state_dict': ...} bundles
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    print(f"✅ Loaded {DEPLOY_CFG['model_name']} from {ckpt}  (T={TEMPERATURE:.4f})")
    return model

MODEL = load_model()

def get_target_layers(model, model_name: str):
    if "resnet" in model_name:
        return [model.layer4[-1]]
    elif "efficientnet" in model_name:
        return [model.blocks[-1]]
    elif "mobilenet" in model_name:
        return [model.blocks[-1]]
    elif "convnext" in model_name:
        return [model.stages[-1]]
    elif "swin" in model_name:
        # Note: attention rollout would be more faithful; this is a known
        # limitation of vanilla Grad-CAM on Swin — documented in report.
        return [model.layers[-1].blocks[-1].norm2]
    return [list(model.children())[-2]]

# ── Calibration via learned temperature scaling ───────────────────────────
def calibrate_prob(fire_logit: float, nf_logit: float) -> float:
    """
    FIX #1 — real temperature scaling.
    Divide both logits by T (learned on val set), then softmax.
    T is read from deploy_config.json; the notebook writes it there.
    If T=1.0 the function is a no-op, preserving backward compat.
    """
    logits = np.array([nf_logit, fire_logit]) / TEMPERATURE
    exp    = np.exp(logits - logits.max())
    return float(exp[1] / exp.sum())   # index 1 is fire after re-sorting

# ── Confidence tiers (honest framing) ─────────────────────────────────────
def confidence_tier(calibrated_fire_prob: float) -> dict:
    """
    FIX #3 — renamed from 'risk score' to 'model confidence score'.
    The UI must label this as 'Model Confidence' not 'Risk Level'.
    """
    score = int(calibrated_fire_prob * 100)
    if calibrated_fire_prob < 0.30:
        tier, color = "Low",      "#22c55e"
    elif calibrated_fire_prob < 0.60:
        tier, color = "Elevated", "#f59e0b"
    else:
        tier, color = "Critical", "#ef4444"
    return {"confidence_score": score, "tier": tier, "color": color}

# ── Grad-CAM ───────────────────────────────────────────────────────────────
def make_gradcam_b64(
    model,
    img_tensor: torch.Tensor,
    rgb_np: np.ndarray,
    predicted_class: int,
) -> str:
    """
    FIX #2 — target the PREDICTED class, not always FIRE_IDX.
    If the model says 'no fire', we explain the no-fire prediction.
    """
    target_layers = get_target_layers(model, DEPLOY_CFG["model_name"])
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale = cam(
        input_tensor=img_tensor.unsqueeze(0).to(DEVICE),
        targets=[ClassifierOutputTarget(predicted_class)],   # ← fixed
    )[0]
    overlay = show_cam_on_image(rgb_np, grayscale, use_rgb=True)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Wildfire Detection API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = TEMPLATE_DIR / "index.html"
    return html_file.read_text(encoding="utf-8")


class PredictionResult(BaseModel):
    prediction        : str
    fire_prob_raw     : float   # uncalibrated softmax
    fire_prob_cal     : float   # temperature-scaled
    confidence_score  : int     # 0-100, model confidence (NOT real-world risk)
    confidence_tier   : str
    confidence_color  : str
    gradcam_b64       : Optional[str]
    gradcam_class     : Optional[str]   # which class the CAM explains
    model_name        : str
    inference_ms      : float   # pure forward-pass time
    gradcam_ms        : Optional[float]
    demo_mode         : bool
    calibration_note  : str     # honest framing shown in UI


@app.post("/predict", response_model=PredictionResult)
async def predict(
    file: UploadFile = File(...),
    include_gradcam: bool = Query(default=True, description="Set false for faster inference"),
):
    # ── FIX #4: input validation ──────────────────────────────────────────
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    raw = await file.read()

    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit")

    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    img_tensor = PREPROCESS(pil_img)
    rgb_np     = INV_NORM(img_tensor).permute(1, 2, 0).numpy().clip(0, 1)

    # ── Inference (timed separately from Grad-CAM) ────────────────────────
    gradcam_b64 = None
    gradcam_ms  = None
    gradcam_cls = None

    if MODEL is None:
        fire_prob_raw = float(np.random.beta(2, 3))
        nf_prob_raw   = 1.0 - fire_prob_raw
        fire_logit    = float(np.log(fire_prob_raw + 1e-8))
        nf_logit      = float(np.log(nf_prob_raw + 1e-8))
        inference_ms  = 0.0
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = MODEL(img_tensor.unsqueeze(0).to(DEVICE))
        inference_ms = (time.perf_counter() - t0) * 1000

        probs         = F.softmax(logits, dim=1)[0]
        fire_prob_raw = float(probs[FIRE_IDX].cpu())
        nf_prob_raw   = float(probs[NO_FIRE_IDX].cpu())
        fire_logit    = float(logits[0][FIRE_IDX].cpu())
        nf_logit      = float(logits[0][NO_FIRE_IDX].cpu())

        # ── Grad-CAM (separate timer, optional) ──────────────────────────
        if include_gradcam:
            predicted_class = FIRE_IDX if fire_prob_raw >= 0.5 else NO_FIRE_IDX
            t1 = time.perf_counter()
            gradcam_b64 = make_gradcam_b64(MODEL, img_tensor, rgb_np, predicted_class)
            gradcam_ms  = round((time.perf_counter() - t1) * 1000, 2)
            gradcam_cls = CLASS_NAMES[predicted_class]

    fire_prob_cal = calibrate_prob(fire_logit, nf_logit)
    pred_label    = "wildfire" if fire_prob_cal >= 0.5 else "no wildfire"
    tier_info     = confidence_tier(fire_prob_cal)

    cal_note = (
        "Temperature-scaled confidence (T learned on validation set). "
        "This reflects model certainty, not real-world fire probability."
        if TEMPERATURE != 1.0
        else "Uncalibrated softmax (no validation checkpoint found). "
             "Treat scores as relative, not absolute probabilities."
    )

    return PredictionResult(
        prediction       = pred_label,
        fire_prob_raw    = round(fire_prob_raw, 4),
        fire_prob_cal    = round(fire_prob_cal, 4),
        confidence_score = tier_info["confidence_score"],
        confidence_tier  = tier_info["tier"],
        confidence_color = tier_info["color"],
        gradcam_b64      = gradcam_b64,
        gradcam_class    = gradcam_cls,
        model_name       = DEPLOY_CFG["model_name"],
        inference_ms     = round(inference_ms, 2),
        gradcam_ms       = gradcam_ms,
        demo_mode        = MODEL is None,
        calibration_note = cal_note,
    )


@app.get("/metrics")
async def benchmark_metrics():
    results_path = MODELS_DIR / "benchmark_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {"message": "No benchmark results found — run the Kaggle notebook first"}


@app.get("/health")
async def health():
    return {
        "status"      : "ok",
        "device"      : str(DEVICE),
        "demo_mode"   : MODEL is None,
        "temperature" : TEMPERATURE,
    }
