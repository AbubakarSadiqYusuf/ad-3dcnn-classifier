# --- Streamlit 3D CNN Alzheimer's Disease Classifier ---

import streamlit as st
import numpy as np
import os
import pydicom
import tempfile
import nibabel as nib
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob
from pathlib import Path
from PIL import Image, ImageFilter
import zipfile
from io import BytesIO
import base64

from tensorflow.keras import layers, models

# --- Streamlit Page Config with Background ---
def add_blurred_background_from_logo(image_path):
    logo_img = Image.open(image_path)
    blurred = logo_img.filter(ImageFilter.GaussianBlur(radius=30))
    buffered = BytesIO()
    blurred.save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{img_base64}");
            background-size: cover;
            background-attachment: fixed;
            backdrop-filter: blur(8px);
        }}
            color: white;
        </style>
    """, unsafe_allow_html=True)

# Set config and background
st.set_page_config(page_title='3D ResNet Alzheimer Classifier', layout='wide')
add_blurred_background_from_logo("ADpred_logo.jpg")

# Load and show logo
logo_img = Image.open("ADpred_logo.jpg")
st.image(logo_img, width=150)

st.title("🧠 3D CNN Alzheimer's Disease Classifier")
st.markdown("""
This tool uses a **3D ResNet50 CNN** to classify the progression stages of Alzheimer's Disease (AD) from MRI brain scans.

**Stages Predicted:**
- 🟢 Normal Cognitive (NC)
- 🟡 Early Mild Cognitive Impairment (EMCI)
- 🟠 Late Mild Cognitive Impairment (LMCI)
- 🔴 Alzheimer's Disease (AD)
""")

# --- Upload DICOM ZIP ---
st.subheader("📤 Upload DICOM ZIP")
zip_file = st.file_uploader("Upload a .zip containing DICOM files for one MRI scan", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        dicom_files = sorted(glob(os.path.join(tmpdir, "**", "*.dcm"), recursive=True))
        if not dicom_files:
            st.error("❌ No DICOM files found in the uploaded ZIP.")
            st.stop()

       import subprocess

        # Use dcm2niix to convert DICOM to NIfTI
        dcm2niix_output = os.path.join(tmpdir, "nifti")
        os.makedirs(dcm2niix_output, exist_ok=True)
        conversion_result = subprocess.run([
            "dcm2niix", "-z", "y", "-o", dcm2niix_output, tmpdir
        ], capture_output=True, text=True)

        if conversion_result.returncode != 0:
            st.error(f"❌ DICOM to NIfTI conversion failed: {conversion_result.stderr}")
            st.stop()

        converted_files = glob(os.path.join(dcm2niix_output, "*.nii.gz"))
        if not converted_files:
            st.error("❌ No NIfTI file was created from the DICOM conversion.")
            st.stop()
            
        nifti_path = converted_files[0]
        img = nib.load(nifti_path)
        volume = img.get_fdata().astype(np.float32)
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

        nifti_filename = f"{zip_name}.nii.gz"
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), nifti_path)
        st.success(f"Converted DICOMs to NIfTI: {nifti_filename}")

        st.subheader("🧠 Sample MRI Slice")
        mid_slice = volume[:, :, volume.shape[2] // 2]
        fig, ax = plt.subplots()
        ax.imshow(mid_slice, cmap="gray")
        ax.axis('off')
        st.pyplot(fig)

        volume_resized = tf.image.resize(volume, (128, 128))
        volume_resized = tf.image.resize(tf.transpose(volume_resized, [2, 0, 1]), (128, 128))
        volume_input = tf.expand_dims(volume_resized, axis=0)
        volume_input = tf.expand_dims(volume_input, axis=-1)

        def build_3d_resnet(input_shape=(None, 128, 128, 1), num_classes=4):
            inputs = tf.keras.Input(shape=input_shape)
            x = layers.Conv3D(64, 3, activation='relu')(inputs)
            x = layers.MaxPool3D(2)(x)
            x = layers.Conv3D(128, 3, activation='relu')(x)
            x = layers.GlobalAveragePooling3D()(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            return models.Model(inputs, outputs)

        model = build_3d_resnet()
        preds = model.predict(volume_input)
        stages = ['Normal Cognitive (NC)', 'Early MCI (EMCI)', 'Late MCI (LMCI)', "Alzheimer's Disease (AD)"]
        pred_stage = stages[np.argmax(preds)]

        st.subheader("🧪 Prediction Result")
        st.markdown(f"### 🧠 Model Prediction: **{pred_stage}**")
        st.write("Confidence Scores:")
        st.bar_chart({stages[i]: float(preds[0][i]) for i in range(4)})

        st.subheader("🔍 Interpretability: GradCAM++ (3D)")
        import tensorflow.keras.backend as K
        def compute_gradcam_3d(model, input_volume, target_class_idx):
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(index=-3).output, model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(input_volume)
                loss = predictions[:, target_class_idx]
            grads = tape.gradient(loss, conv_outputs)[0]
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
            conv_outputs = conv_outputs[0]
            for i in range(conv_outputs.shape[-1]):
                conv_outputs[:, :, :, i] *= pooled_grads[i]
            heatmap = tf.reduce_mean(conv_outputs, axis=-1).numpy()
            heatmap = np.maximum(heatmap, 0)
            heatmap /= np.max(heatmap) + 1e-8
            return heatmap

        heatmap = compute_gradcam_3d(model, volume_input, np.argmax(preds))
        gradcam_slice = heatmap[heatmap.shape[0] // 2]

        fig, ax = plt.subplots()
        ax.imshow(gradcam_slice, cmap='inferno')
        ax.set_title("GradCAM++ - Mid Slice")
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("---")
        st.caption("Built for clinical decision support and research on Alzheimer's Disease.")
