import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import joblib
from PIL import Image

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
# 2. MODEL LOADING
# ==========================================
def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

@st.cache_resource
def load_all_models():
    try:
        seg_model = load_model(
            'thyroid_final_corrected.keras',
            custom_objects={'dice_loss': dice_loss, 'dice_coef': dice_coef}
        )
    except:
        st.error("Could not load segmentation model.")
        return None, None, None

    try:
        rf_model = joblib.load('thyroid_rf_classifier.pkl')
    except:
        st.error("Could not load Random Forest classifier.")
        return None, None, None

    vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg',
                      input_shape=(128, 128, 3))

    return seg_model, rf_model, vgg_model

seg_model, rf_model, vgg_model = load_all_models()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def create_heatmap(image, mask_prob):
    mask_resized = cv2.resize(mask_prob, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * mask_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(heatmap, 0.4, image, 0.6, 0)
    return overlay

def process_image(image_file):
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def diagnose(img_arr):
    img_seg_in = cv2.resize(img_arr, (256, 256))
    img_seg_norm = img_seg_in / 255.0
    img_seg_batch = np.expand_dims(img_seg_norm, axis=0)

    pred_mask = seg_model.predict(img_seg_batch, verbose=0)[0]

    threshold = 0.3 if pred_mask.max() > 0.3 else 0.5
    pred_mask_bin = (pred_mask > threshold).astype(np.uint8)

    heatmap_img = create_heatmap(img_seg_in, pred_mask)

    contours, _ = cv2.findContours(
        pred_mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return img_seg_in, heatmap_img, None, "No Nodule Detected", 0.0, "#000000"

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    pad = 10
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(256 - x, w + 2 * pad); h = min(256 - y, h + 2 * pad)
    crop = img_seg_in[y:y+h, x:x+w]

    crop_vgg = cv2.resize(crop, (128, 128))
    crop_batch = np.expand_dims(crop_vgg, axis=0)
    crop_pre = preprocess_input(crop_batch.copy())

    features = vgg_model.predict(crop_pre, verbose=0)
    probs = rf_model.predict_proba(features)[0]

    if probs[2] > 0.25:
        diag = "Malignant"
        conf = probs[2]
        color = "#ff4b4b"
    elif probs[1] > probs[0]:
        diag = "Suspicious"
        conf = probs[1]
        color = "#ffa500"
    else:
        diag = "Benign"
        conf = probs[0]
        color = "#4caf50"

    return img_seg_in, heatmap_img, crop_vgg, diag, conf, color

# ==========================================
# 4. MAIN UI
# ==========================================
st.title("🏥 Thyroid Ultrasound AI Diagnostic System")
st.write("Upload an ultrasound image to detect nodules and classify malignancy risk.")

uploaded_file = st.file_uploader("Choose an Ultrasound Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner('Analyzing Image...'):
        original_img = process_image(uploaded_file)
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
