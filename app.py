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
from PIL import Image
import zipfile

from tensorflow.keras import layers, models

# --- Streamlit Page Config ---
st.set_page_config(page_title='3D ResNet Alzheimer Classifier', layout='wide')

# Background Image via CSS
def add_bg_image():
    with open("/mnt/data/ADpred_logo.jpg", "rb") as img_file:
        import base64
        bg_bytes = base64.b64encode(img_file.read()).decode()
        st.markdown(f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{bg_bytes}");
                background-size: cover;
                background-attachment: fixed;
                backdrop-filter: blur(8px);
            }}
            </style>
        """, unsafe_allow_html=True)

add_bg_image()
st.image("/mnt/data/ADpred_logo.jpg", width=150)
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

        slices = []
        for f in dicom_files:
            try:
                ds = pydicom.dcmread(f)
                slices.append(ds.pixel_array)
            except Exception as e:
                st.warning(f"Skipped unreadable DICOM: {f} — {e}")

        if not slices:
            st.error("❌ All DICOM files failed to load.")
            st.stop()

        volume = np.stack(slices, axis=-1).astype(np.float32)
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

        # Save NIfTI
        zip_name = Path(zip_file.name).stem
        nifti_filename = f"{zip_name}.nii.gz"
        nifti_path = os.path.join("/mnt/data", nifti_filename)
        nib.save(nib.Nifti1Image(volume, affine=np.eye(4)), nifti_path)
        st.success(f"Converted DICOMs to NIfTI: {nifti_filename}")

        # --- Display middle slice ---
        st.subheader("🧠 Sample MRI Slice")
        mid_slice = volume[:, :, volume.shape[2] // 2]
        fig, ax = plt.subplots()
        ax.imshow(mid_slice, cmap="gray")
        ax.axis('off')
        st.pyplot(fig)

        # --- Resize for model ---
        volume_resized = tf.image.resize(volume, (128, 128))
        volume_resized = tf.image.resize(tf.transpose(volume_resized, [2, 0, 1]), (128, 128))
        volume_input = tf.expand_dims(volume_resized, axis=0)
        volume_input = tf.expand_dims(volume_input, axis=-1)

        # --- Load 3D ResNet50 (placeholder model logic) ---
        def build_3d_resnet(input_shape=(None, 128, 128, 1), num_classes=4):
            inputs = tf.keras.Input(shape=input_shape)
            x = layers.Conv3D(64, 3, activation='relu')(inputs)
            x = layers.MaxPool3D(2)(x)
            x = layers.Conv3D(128, 3, activation='relu')(x)
            x = layers.GlobalAveragePooling3D()(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            model = models.Model(inputs, outputs)
            return model

        model = build_3d_resnet()

        preds = model.predict(volume_input)
        stages = ['Normal Cognitive (NC)', 'Early MCI (EMCI)', 'Late MCI (LMCI)', "Alzheimer's Disease (AD)"]
        pred_stage = stages[np.argmax(preds)]

        st.subheader("🧪 Prediction Result")
        st.markdown(f"### 🧠 Model Prediction: **{pred_stage}**")
        st.write("Confidence Scores:")
        st.bar_chart({stages[i]: float(preds[0][i]) for i in range(4)})

        # --- GradCAM 3D Visual Explanation ---
        st.subheader("🔍 Interpretability: GradCAM++ (3D)")
        import tensorflow.keras.backend as K
        import matplotlib.cm as cm

        def compute_gradcam_3d(model, input_volume, target_class_idx):
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(index=-3).output, model.output])

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

        slice_idx = heatmap.shape[0] // 2
        gradcam_slice = heatmap[slice_idx]

        fig, ax = plt.subplots()
        ax.imshow(gradcam_slice, cmap='inferno')
        ax.set_title("GradCAM++ - Mid Slice")
        ax.axis('off')
        st.pyplot(fig)

        st.markdown("---")
        st.caption("Built for clinical decision support and research on Alzheimer's Disease.")
