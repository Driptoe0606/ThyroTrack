# app.py (PyTorch + Pillow version)
import os
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.cm as cm
from scipy import ndimage

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="Thyroid AI Diagnosis", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .diagnosis-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. MODEL LOADING (PyTorch + joblib)
# ==========================================
@st.cache_resource(show_spinner=False)
def load_all_models(device="cpu"):
    # Segmentation model (PyTorch .pth)
    seg_model = None
    rf_model = None
    vgg_feat = None

    # Load segmentation model (assumed to be a PyTorch state_dict or scripted module)
    seg_path = "app_folder/best_unet.pth"
    if os.path.exists(seg_path):
        try:
            # We try several common load patterns:
            # 1) scripted/traced module saved with torch.jit.save
            # 2) state_dict for a common UNet class (user must have saved state_dict)
            # 3) full torch.save(module)
            try:
                seg_model = torch.jit.load(seg_path, map_location=device)
            except Exception:
                # try loading as state dict into a generic small UNet skeleton if needed
                # If you have a custom UNet class, replace this block by your model class.
                # We'll attempt to load as a full object first:
                loaded = torch.load(seg_path, map_location=device)
                if isinstance(loaded, dict) and "state_dict" in loaded:
                    # saved checkpoint with "state_dict"
                    state = loaded["state_dict"]
                    # user must supply UNet class if needed; fallback: try load as module
                    seg_model = torch.nn.Module()
                    seg_model.load_state_dict(state)  # may fail if shape mismatch
                else:
                    # loaded is likely a module
                    seg_model = loaded
            seg_model.to(device)
            seg_model.eval()
        except Exception as e:
            st.error(f"Could not load segmentation model: {e}")
            seg_model = None
    else:
        st.error("Segmentation model file not found (app_folder/best_unet.pth).")

    # Load Random Forest classifier (joblib)
    rf_path = "app_folder/thyroid_rf_classifier.pkl"
    if os.path.exists(rf_path):
        try:
            rf_model = joblib.load(rf_path)
        except Exception as e:
            st.error(f"Could not load Random Forest classifier: {e}")
            rf_model = None
    else:
        st.error("Random Forest classifier file not found (app_folder/thyroid_rf_classifier.pkl).")

    # Load VGG16 feature extractor (torchvision)
    try:
        # create a feature extractor that returns a 512-d vector (avg pooled)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feat_extractor = nn.Sequential(*list(vgg.features.children()),
                                       nn.AdaptiveAvgPool2d((1, 1)))
        # Add a flatten wrapper
        class FeatWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module
            def forward(self, x):
                x = self.module(x)
                return x.view(x.size(0), -1)
        vgg_feat = FeatWrapper(feat_extractor).to(device)
        vgg_feat.eval()
    except Exception as e:
        st.error(f"Could not initialize VGG feature extractor: {e}")
        vgg_feat = None

    return seg_model, rf_model, vgg_feat

device = "cpu"  # Streamlit Cloud typically uses CPU-only environment
seg_model, rf_model, vgg_feat = load_all_models(device=device)

# ==========================================
# 3. HELPERS (Pillow + NumPy replacements for OpenCV)
# ==========================================
def pil_to_np(img_pil):
    """Convert PIL RGB image to numpy uint8 array (H,W,3)."""
    return np.asarray(img_pil.convert("RGB"))

def np_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

def resize_np_image(img_np, size):
    pil = np_to_pil(img_np)
    pil = pil.resize(size, resample=Image.BILINEAR)
    return np.asarray(pil)

def create_heatmap(image_np, mask_prob):
    """
    image_np: HxWx3 uint8 RGB
    mask_prob: HxW float (0-1)
    returns: overlay uint8 RGB of same shape as image_np
    """
    # Resize mask to image size
    mask_pil = Image.fromarray((mask_prob * 255).astype(np.uint8))
    mask_resized = np.asarray(mask_pil.resize((image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR)) / 255.0
    cmap = cm.get_cmap("jet")
    colored = (cmap(mask_resized)[..., :3] * 255).astype(np.uint8)  # HxWx3
    # Blend: overlay = 0.4*heatmap + 0.6*image
    overlay = (0.4 * colored.astype(np.float32) + 0.6 * image_np.astype(np.float32)).astype(np.uint8)
    return overlay

def process_image_pil(uploaded_file):
    """Return numpy RGB image from uploaded file"""
    img_pil = Image.open(uploaded_file).convert("RGB")
    return pil_to_np(img_pil)

def get_connected_component_bbox(binary_mask):
    """Return bounding box x,y,w,h of the largest connected component. binary_mask is HxW 0/1"""
    labeled, n = ndimage.label(binary_mask)
    if n == 0:
        return None
    # find largest component by area excluding label 0
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = counts.argmax()
    coords = np.where(labeled == largest_label)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    x, y, w, h = int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)
    return x, y, w, h

# Preprocessing transform for VGG (128x128)
vgg_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. DIAGNOSE FUNCTION (uses PyTorch models)
# ==========================================
def diagnose(img_arr):
    """
    img_arr: HxWx3 uint8 RGB numpy array
    returns: resized_orig, heatmap, crop_vgg (128x128 numpy), diagnosis, confidence, color
    """
    if seg_model is None or rf_model is None or vgg_feat is None:
        st.error("One or more models failed to load.")
        return img_arr, img_arr, None, "Model Error", 0.0, "#000000"

    # Resize for segmentation input (256x256)
    seg_input = resize_np_image(img_arr, (256, 256)).astype(np.float32) / 255.0
    seg_tensor = torch.from_numpy(seg_input.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        try:
            out = seg_model(seg_tensor)
        except Exception as e:
            # try common pattern where model returns dict or tuple
            try:
                out = seg_model(seg_tensor)[0]
            except Exception as e2:
                st.error(f"Segmentation model inference failed: {e}; {e2}")
                return img_arr, img_arr, None, "Segmentation Error", 0.0, "#000000"

    # assume model output is logits or probabilities single-channel
    if isinstance(out, torch.Tensor):
        pred = out.squeeze().cpu()
    elif isinstance(out, (list, tuple)):
        pred = out[0].squeeze().cpu()
    else:
        pred = torch.tensor(out).squeeze()

    # If logits (values outside 0-1), apply sigmoid
    pred_np = pred.numpy()
    if pred_np.max() > 1.0 or pred_np.min() < 0.0:
        pred_np = 1.0 / (1.0 + np.exp(-pred_np))  # sigmoid

    # Binarize mask using adaptive threshold rule
    threshold = 0.3 if pred_np.max() > 0.3 else 0.5
    pred_bin = (pred_np > threshold).astype(np.uint8)

    # Create heatmap overlay on resized seg_input (converted back to uint8)
    seg_input_uint8 = (seg_input * 255).astype(np.uint8)
    heatmap_img = create_heatmap(seg_input_uint8, pred_np)

    # Find largest connected component bounding box
    bbox = get_connected_component_bbox(pred_bin)
    if bbox is None:
        return seg_input_uint8, heatmap_img, None, "No Nodule Detected", 0.0, "#000000"

    x, y, w, h = bbox
    # pad and clamp
    pad = 10
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(256 - x, w + 2 * pad); h = min(256 - y, h + 2 * pad)
    crop = seg_input_uint8[y:y+h, x:x+w]

    # prepare crop for VGG feature extractor
    crop_pil = Image.fromarray(crop)
    crop_tensor = vgg_transform(np.asarray(crop_pil)).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = vgg_feat(crop_tensor)  # shape (1, 512)
    feats_np = feats.cpu().numpy().reshape(1, -1)

    try:
        probs = rf_model.predict_proba(feats_np)[0]
    except Exception as e:
        st.error(f"Classifier predict_proba failed: {e}")
        return seg_input_uint8, heatmap_img, np.asarray(crop_pil.resize((128,128))), "Classifier Error", 0.0, "#000000"

    # Interpret probabilities (assumes rf_model classes are [benign, suspicious, malignant])
    # If class ordering is different, adjust accordingly.
    if len(probs) >= 3:
        if probs[2] > 0.25:
            diag = "Malignant"
            conf = float(probs[2])
            color = "#ff4b4b"
        elif probs[1] > probs[0]:
            diag = "Suspicious"
            conf = float(probs[1])
            color = "#ffa500"
        else:
            diag = "Benign"
            conf = float(probs[0])
            color = "#4caf50"
    else:
        # fallback: take top class
        top_idx = int(np.argmax(probs))
        conf = float(probs[top_idx])
        diag = str(rf_model.classes_[top_idx]) if hasattr(rf_model, "classes_") else f"Class {top_idx}"
        color = "#4caf50" if "Benign" in diag else "#ffa500" if "Suspicious" in diag else "#ff4b4b"

    crop_vgg = np.asarray(crop_pil.resize((128, 128)))
    return seg_input_uint8, heatmap_img, crop_vgg, diag, conf, color

# ==========================================
# 5. MAIN UI
# ==========================================
st.title("🏥 Thyroid Ultrasound AI Diagnostic System")
st.write("Upload an ultrasound image to detect nodules and classify malignancy risk.")

uploaded_file = st.file_uploader("Choose an Ultrasound Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Analyzing Image...'):
        original_img = process_image_pil(uploaded_file)
        resized_orig, heatmap, crop, diagnosis, confidence, color = diagnose(original_img)

    st.markdown(f"""
    <div class="diagnosis-box" style="background-color: {color}20; border: 2px solid {color};">
        <h2 style="color: {color}; margin:0;">Prediction: {diagnosis}</h2>
        <p style="margin:0;">Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(resized_orig, caption="Original Scan (Resized)", use_column_width=True)

    with col2:
        st.image(heatmap, caption="AI Segmentation Heatmap", use_column_width=True)

    with col3:
        if crop is not None:
            st.image(crop, caption="Auto-Cropped Nodule", width=150)
        else:
            st.warning("No Nodule Found")

    st.info(
        "Red regions in the heatmap indicate high AI confidence for nodule presence. "
        "The crop shows the area sent to the classifier."
    )
