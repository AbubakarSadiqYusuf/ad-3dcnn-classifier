# Alzheimer's Disease 3D CNN Classifier

This Streamlit app uses a 3D ResNet-based CNN to classify brain MRI DICOM volumes into one of four Alzheimer's disease progression stages:

- Normal Cognitive (NC)
- Early Mild Cognitive Impairment (EMCI)
- Late Mild Cognitive Impairment (LMCI)
- Alzheimer's Disease (AD)

It includes a GradCAM++-based 3D interpretability module for visual explanation of the model's predictions.

## Getting Started

1. Clone this repo or upload it to Streamlit Cloud.
2. Upload a ZIP file containing a DICOM series.
3. View predictions and interpretability results.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
