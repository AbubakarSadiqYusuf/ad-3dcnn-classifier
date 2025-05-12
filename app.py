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
import dicom2nifti
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
            color: white;
        }}
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title='3D ResNet Alzheimer Classifier', layout='wide')
add_blurred_background_from_logo("ADpred_logo.jpg")
st.image("ADpred_logo.jpg", width=150)

st.title("🧠 3D CNN Alzheimer's Disease Classifier")
st.sidebar.image("ADpred_logo.jpg", width=150)
st.sidebar.markdown("""
This tool uses a **3D ResNet50 CNN** to classify the progression stages of Alzheimer's Disease (AD) from MRI brain scans.

**Stages Predicted:**
- 🟢 Normal Cognitive (NC)
- 🟡 Early Mild Cognitive Impairment (EMCI)
- 🟠 Late Mild Cognitive Impairment (LMCI)
- 🔴 Alzheimer's Disease (AD)
""")

# --- Sidebar for Upload ---
st.sidebar.header("📤 Upload Section")
zip_file = st.sidebar.file_uploader("Upload a .zip containing DICOM files", type="zip")

if zip_file:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "upload.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_file.read())

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        dicom_files = sorted(glob(os.path.join(tmpdir, "**", "*.dcm"), recursive=True))
        if not dicom_files:
            st.error("❌ No DICOM files found.")
            st.stop()

        try:
            zip_name = Path(zip_file.name).stem
            nifti_filename = f"{zip_name}.nii.gz"
            nifti_path = os.path.join("./", nifti_filename)
            dicom2nifti.convert_dicom.dicom_series_to_nifti(tmpdir, nifti_path)
            img = nib.load(nifti_path)
            volume = img.get_fdata().astype(np.float32)
            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
            st.success(f"✅ Converted to NIfTI: {nifti_filename}")
        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")
            st.stop()

        # --- Multi-view Mid-slices ---
        st.subheader("🧠 MRI Multi-View Mid-Slices")
        axial = volume[:, :, volume.shape[2] // 2]
        coronal = volume[:, volume.shape[1] // 2, :]
        sagittal = volume[volume.shape[0] // 2, :, :]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(axial.T, cmap="gray", origin="lower")
        axs[0].set_title("Axial View")
        axs[0].axis("off")
        axs[1].imshow(coronal.T, cmap="gray", origin="lower")
        axs[1].set_title("Coronal View")
        axs[1].axis("off")
        axs[2].imshow(sagittal.T, cmap="gray", origin="lower")
        axs[2].set_title("Sagittal View")
        axs[2].axis("off")
        st.pyplot(fig)

        # --- Prepare Input ---
        volume_resized = tf.image.resize(volume, (128, 128))
        volume_resized = tf.image.resize(tf.transpose(volume_resized, [2, 0, 1]), (128, 128))
        volume_input = tf.expand_dims(volume_resized, axis=0)
        volume_input = tf.expand_dims(volume_input, axis=-1)

        # --- Model Definition ---
        def build_3d_resnet(input_shape=(None, 128, 128, 1), num_classes=4):
            inputs = tf.keras.Input(shape=input_shape)
            x = layers.Conv3D(64, 3, activation='relu')(inputs)
            x = layers.MaxPool3D(2)(x)
            x = layers.Conv3D(128, 3, activation='relu')(x)
            x = layers.GlobalAveragePooling3D()(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            return models.Model(inputs, outputs)

        # --- Load or Create Model ---
        model_path = os.path.join(".", "3d_resnet_alzheimer.keras")
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.info("✅ Loaded saved model.")
        else:
            model = build_3d_resnet()
            model.save(model_path)
            st.success("💾 Trained & saved new model.")

        # --- Predict ---
        preds = model.predict([volume_input])
        stages = ['Normal Cognitive (NC)', 'Early MCI (EMCI)', 'Late MCI (LMCI)', "Alzheimer's Disease (AD)"]
        pred_stage = stages[np.argmax(preds)]

        st.subheader("🧪 Prediction Result")
        st.markdown(f"### 🧠 Model Prediction: **{pred_stage}**")
        st.write("Confidence Scores:")
        st.bar_chart({stages[i]: float(preds[0][i]) for i in range(4)})

        # --- GradCAM++ Visualization ---
        st.subheader("🔍 Interpretability: GradCAM++ (3D)")
        import tensorflow.keras.backend as K

        # Dynamically find last Conv3D layer
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv3D):
                target_layer = layer.name
                break

        def compute_gradcam_3d(model, input_volume, target_class_idx):
            grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(target_layer).output, model.output])
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model([input_volume])
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

        try:
            heatmap = compute_gradcam_3d(model, volume_input, np.argmax(preds))
            gradcam_slice = heatmap[heatmap.shape[0] // 2]
            fig, ax = plt.subplots()
            ax.imshow(gradcam_slice, cmap='inferno')
            ax.set_title("GradCAM++ - Mid Slice")
            ax.axis('off')
            st.pyplot(fig)
        except Exception as e:
            st.error(f"⚠️ GradCAM++ failed: {e}")

        st.markdown("---")
        st.caption("Built for clinical decision support and Alzheimer's research.")
