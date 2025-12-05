# app.py (PyTorch + Pillow version)
import os
import streamlit as st
import numpy as np
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from scipy import ndimage
import matplotlib.cm as cm

# ==========================================
# 1. CONFIG & STYLES
# ==========================================
st.set_page_config(page_title="Thyroid AI Diagnosis", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; background-color: #ff4b4b; color: white; }
    .diagnosis-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL DEFINITIONS
# ==========================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=(32,64,128,256)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f
        self.pool = nn.MaxPool2d(2,2)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        rev = list(reversed(features))
        up_in = features[-1]*2
        for f in rev:
            self.ups.append(nn.ConvTranspose2d(up_in, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(up_in, f))
            up_in = f
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)
        
    def forward(self, x):
        skips = []
        out = x
        for d in self.downs:
            out = d(out)
            skips.append(out)
            out = self.pool(out)
        out = self.bottleneck(out)
        skips = skips[::-1]
        idx = 0
        for i in range(0, len(self.ups), 2):
            trans = self.ups[i]
            conv = self.ups[i+1]
            out = trans(out)
            skip = skips[idx]; idx += 1
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:])
            out = torch.cat([skip, out], dim=1)
            out = conv(out)
        return self.final(out)

# ==========================================
# 3. MODEL LOADING FUNCTION
# ==========================================
@st.cache_resource(show_spinner=False)
def load_all_models(device="cpu"):
    seg_model, rf_model, vgg_feat = None, None, None

    # ---- 1) Segmentation model ----
    seg_path = "app_folder/best_unet.pth"
    if os.path.exists(seg_path):
        try:
            seg_model = UNet().to(device)
            checkpoint = torch.load("app_folder/best_unet.pth", map_location="cpu")
            print(checkpoint.keys())
            # If checkpoint contains model_state
            state_dict = checkpoint.get("model_state", checkpoint)
            print(list(state_dict.keys())[:10])
            
            # Check if it's a full checkpoint
            if isinstance(checkpoint, dict):
                if "model_state" in checkpoint:
                    state_dict = checkpoint["model_state"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    # fallback: assume whole dict is state_dict
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
    
            seg_model = UNet(features=(32,64,128,256), out_channels=1).to(device)
            seg_model.load_state_dict(checkpoint["model_state"])
            seg_model.load_state_dict(state_dict)
            seg_model.eval()
            st.info("Segmentation model loaded successfully.")
        except Exception as e:
            st.error(f"Failed to load UNet: {e}")
            seg_model = None
    else:
        st.error(f"Segmentation model file not found at: {seg_path}")

    # ---- 2) Random Forest ----
    rf_path = "app_folder/thyroid_rf_classifier.pkl"
    if os.path.exists(rf_path):
        try:
            rf_model = joblib.load(rf_path)
        except Exception as e:
            st.error(f"Could not load Random Forest classifier: {e}")
            rf_model = None
    else:
        st.error(f"Random Forest classifier file not found at: {rf_path}")

    # ---- 3) VGG16 feature extractor ----
    try:
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        feat_extractor = nn.Sequential(
            *list(vgg.features.children()),
            nn.AdaptiveAvgPool2d((1,1))
        )
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

# ==========================================
# 4. IMAGE HELPERS (Pillow + NumPy)
# ==========================================
def pil_to_np(img_pil):
    return np.asarray(img_pil.convert("RGB"))

def np_to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

def resize_np_image(img_np, size):
    pil = np_to_pil(img_np)
    pil = pil.resize(size, resample=Image.BILINEAR)
    return np.asarray(pil)

def create_heatmap(image_np, mask_prob):
    mask_pil = Image.fromarray((mask_prob*255).astype(np.uint8))
    mask_resized = np.asarray(mask_pil.resize((image_np.shape[1], image_np.shape[0]), resample=Image.BILINEAR))/255.0
    cmap = cm.get_cmap("jet")
    colored = (cmap(mask_resized)[...,:3]*255).astype(np.uint8)
    overlay = (0.4*colored.astype(np.float32) + 0.6*image_np.astype(np.float32)).astype(np.uint8)
    return overlay

def process_image_pil(uploaded_file):
    img_pil = Image.open(uploaded_file).convert("RGB")
    return pil_to_np(img_pil)

def get_connected_component_bbox(binary_mask):
    labeled, n = ndimage.label(binary_mask)
    if n == 0: return None
    counts = np.bincount(labeled.ravel()); counts[0]=0
    largest_label = counts.argmax()
    coords = np.where(labeled==largest_label)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return int(x_min), int(y_min), int(x_max-x_min+1), int(y_max-y_min+1)

# VGG preprocessing
vgg_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

# ==========================================
# 5. DIAGNOSE FUNCTION
# ==========================================
def diagnose(img_arr, seg_model, rf_model, vgg_feat, device="cpu"):
    if seg_model is None or rf_model is None or vgg_feat is None:
        st.error("One or more models failed to load.")
        return img_arr, img_arr, None, "Model Error", 0.0, "#000000"

    # Segmentation input
    seg_input = resize_np_image(img_arr, (256,256)).astype(np.float32)/255.0
    seg_tensor = torch.from_numpy(seg_input.transpose(2,0,1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = seg_model(seg_tensor)
        pred = out.squeeze().cpu().numpy()
        if pred.max()>1.0 or pred.min()<0.0:
            pred = 1.0/(1.0+np.exp(-pred))
    pred_bin = (pred>0.3).astype(np.uint8)
    heatmap_img = create_heatmap((seg_input*255).astype(np.uint8), pred)

    bbox = get_connected_component_bbox(pred_bin)
    if bbox is None:
        return (seg_input*255).astype(np.uint8), heatmap_img, None, "No Nodule Detected", 0.0, "#000000"

    x,y,w,h = bbox; pad=10
    x,y = max(0,x-pad), max(0,y-pad)
    w,h = min(256-x, w+2*pad), min(256-y, h+2*pad)
    crop = seg_input[y:y+h, x:x+w]
    crop_pil = Image.fromarray((crop*255).astype(np.uint8))
    crop_tensor = vgg_transform(np.asarray(crop_pil)).unsqueeze(0).to(device)
    with torch.no_grad():
        feats = vgg_feat(crop_tensor)
    feats_np = feats.cpu().numpy().reshape(1,-1)
    try:
        probs = rf_model.predict_proba(feats_np)[0]
    except:
        return (seg_input*255).astype(np.uint8), heatmap_img, np.asarray(crop_pil.resize((128,128))), "Classifier Error", 0.0, "#000000"

    if len(probs)>=3:
        if probs[2]>0.25: diag,conf,color="Malignant",float(probs[2]),"#ff4b4b"
        elif probs[1]>probs[0]: diag,conf,color="Suspicious",float(probs[1]),"#ffa500"
        else: diag,conf,color="Benign",float(probs[0]),"#4caf50"
    else:
        top_idx=int(np.argmax(probs))
        diag=str(rf_model.classes_[top_idx]) if hasattr(rf_model,"classes_") else f"Class {top_idx}"
        conf=float(probs[top_idx])
        color="#4caf50" if "Benign" in diag else "#ffa500" if "Suspicious" in diag else "#ff4b4b"

    return (seg_input*255).astype(np.uint8), heatmap_img, np.asarray(crop_pil.resize((128,128))), diag, conf, color

# ==========================================
# 6. MAIN UI
# ==========================================
st.title("🏥 Thyroid Ultrasound AI Diagnostic System")
st.write("Upload an ultrasound image to detect nodules and classify malignancy risk.")

device = "cuda" if torch.cuda.is_available() else "cpu"
seg_model, rf_model, vgg_feat = load_all_models(device=device)

uploaded_file = st.file_uploader("Choose an Ultrasound Image...", type=["jpg","jpeg","png"])

if uploaded_file:
    with st.spinner('Analyzing Image...'):
        original_img = process_image_pil(uploaded_file)
        resized_orig, heatmap, crop, diagnosis, confidence, color = diagnose(
            original_img, seg_model, rf_model, vgg_feat, device=device
        )

    st.markdown(f"""
    <div class="diagnosis-box" style="background-color: {color}20; border: 2px solid {color};">
        <h2 style="color: {color}; margin:0;">Prediction: {diagnosis}</h2>
        <p style="margin:0;">Confidence: {confidence*100:.1f}%</p>
    </div>
    """, unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    col1.image(resized_orig, caption="Original Scan (Resized)", use_column_width=True)
    col2.image(heatmap, caption="AI Segmentation Heatmap", use_column_width=True)
    col3.image(crop, caption="Auto-Cropped Nodule", width=150) if crop is not None else col3.warning("No Nodule Found")

    st.info("Red regions indicate high AI confidence. Crop shows area sent to classifier.")
