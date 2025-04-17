import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
from datetime import datetime, date
from fpdf import FPDF
import io
import os
import qrcode
import gdown

# ----------------- Google Drive File IDs -----------------
file_ids = {
    "image_classification_with_augmentation.h5": "1Mx4aIan92NRkJCttQTxC-mHo50dHnm36",
    "fine_tuned_vgg16.h5": "1CqRq_vVArS4jJdt3W5xvgJIYDsWqrjQP",
    "r_model.h5": "1kAfYa6C3XfJ2Yy3PFE4gnNxncEuNaBSh",
    "g_model.h5": "12bjK1s3HBnUGfdoB5IY1TsIMtJK6QjSc",
    "b_model.h5": "10ebJDDODwZr8a4Zh4WishB8PZ5IeVM8J"
}

# ----------------- Setup Models -----------------
models = {
    "CNN Model": "image_classification_with_augmentation.h5",
    "Transfer Learning Model": "fine_tuned_vgg16.h5",
    "Multi-Channel Model": ["r_model.h5", "g_model.h5", "b_model.h5"],
}

# ----------------- Download from Google Drive -----------------
@st.cache_data
def download_model_files():
    for model_value in models.values():
        paths = model_value if isinstance(model_value, list) else [model_value]
        for filename in paths:
            if not os.path.exists(filename):
                file_id = file_ids.get(filename)
                if file_id:
                    url = f"https://drive.google.com/uc?id={file_id}"
                    st.write(f"Downloading {filename}...")
                    gdown.download(url, filename, quiet=False)
                else:
                    st.error(f"File ID not found for {filename}")

download_model_files()

# ----------------- Load Models -----------------
@st.cache_resource
def load_model(model_path):
    if isinstance(model_path, list):
        return [tf.keras.models.load_model(path) for path in model_path]
    return tf.keras.models.load_model(model_path)

# ----------------- Preprocessing Functions -----------------
def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_image_channels(image, target_size):
    img = image.resize(target_size).convert("RGB")
    img_array = np.array(img) / 255.0
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    return (
        np.expand_dims(r, axis=(-1, 0)),
        np.expand_dims(g, axis=(-1, 0)),
        np.expand_dims(b, axis=(-1, 0)),
    )

def predict_image_channels(image, models, target_size):
    r_model, g_model, b_model = models
    r_channel, g_channel, b_channel = preprocess_image_channels(image, target_size)
    r_pred = r_model.predict(r_channel)
    g_pred = g_model.predict(g_channel)
    b_pred = b_model.predict(b_channel)
    combined_predictions = np.array([r_pred, g_pred, b_pred]).sum(axis=0)
    final_prediction = np.argmax(combined_predictions)
    return final_prediction, combined_predictions[0][final_prediction]

# ----------------- Streamlit App -----------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #e6f2fb !important;
        }
        .block-container {
            padding-top: 6rem !important;
            padding-bottom: 2rem !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
        }
        .st-b7{
            background-color: white;
            
        }
        .st-bc {
        color: black;
        }
        .main-title {
            font-size: 2.8em;
            font-weight: 800;
            text-align: center;
            color: #0056b3;
            margin-top: 0rem;
            margin-bottom: 0.25em;
        }
        .stMarkdown,p, .st-emotion-cache-13k62yr{
            color: black;
        }
        .st-emotion-cache-b0y9n5, .st-emotion-cache-jh76sn{
            background-color: #458FF6;
        }
        .st-emotion-cache-1erivf3, .st-de, .st-emotion-cache-xwtqgq{
            background-color: #9bd2f9;
            border: none;
        }
        .sub-title {
            font-size: 1.2em;
            text-align: center;
            color: #222222;
            margin-bottom: 2.5em;
        }
        .report-box {
            background-color: white;
            color: black;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
        }
        .styled-table {
            background-color: #ffffff;
            color: #111;
            border-collapse: collapse;
            margin: 20px auto;
            font-size: 14px;
            width: 90%;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .styled-table th, .styled-table td {
            border: 1px solid #dfe6ed;
            padding: 10px 12px;
            text-align: center;
        }
        .styled-table thead {
            background-color: #007bff;
            color: white;
        }
        .styled-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .styled-table tr:hover {
            background-color: #edf2fa;
        }
        .plot-container {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.07);
            margin-top: 20px;
        }
        section[data-testid="stSidebar"] {
            background-color: #f1f5f9 !important;
            color: black !important;
        }
        section[data-testid="stSidebar"] label {
            color: #000000 !important;
            font-weight: 600 !important;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox div {


        border: 1px solid #ccc !important;
            color: #000 !important;
            background-color: #fff !important;
        }
        .stDownloadButton > button {
            background-color: #1d8cf8;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 6px;
            border: none;
        }
        .stFileUploader {
            margin-top: 1rem;
            margin-bottom: 2rem;
            padding: 1rem;
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        }
        .stFileUploader label {
            color: #000 !important;
            font-weight: 600 !important;
            font-size: 1rem;
            margin-bottom: 0.5rem !important;
            display: block;
        }
        .stFileUploader div[data-testid="stFileDropzone"] {
            border: 2px dashed #007bff !important;
            background-color: #f0f8ff !important;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
        .stFileUploader .uploadedFile {
            background-color: #e0f3ff;
            color: #0056b3;
            border-radius: 15px;
            padding: 5px 12px;
            margin-top: 10px;
            display: inline-block;
            font-weight: 500;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🩺 UCA Clinic Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Accurate, AI-assisted Leishmaniasis Detection</div>', unsafe_allow_html=True)






# Sidebar Inputs
st.sidebar.header("👤 Patient Info")
patient_name = st.sidebar.text_input("Full Name")
patient_age = st.sidebar.number_input("Age", min_value=0, max_value=120, step=1)
patient_gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
patient_dob = st.sidebar.date_input("Date of Birth", min_value=date(1900, 1, 1), max_value=date.today())

st.sidebar.markdown("---")
uploaded_logo = st.sidebar.file_uploader("Upload Hospital/Clinic Logo (optional)", type=["png", "jpg", "jpeg"])
logo_path = None
if uploaded_logo:
    logo_path = f"uploaded_{uploaded_logo.name}"
    with open(logo_path, "wb") as f:
        f.write(uploaded_logo.getbuffer())

st.sidebar.markdown("---")
selected_model = st.sidebar.selectbox("Choose a model:", list(models.keys()))
model_path = models[selected_model]
model = load_model(model_path)

uploaded_files = st.file_uploader("📤 Upload Image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# ----------------- PDF Class -----------------
class PDF(FPDF):
    def header(self):
        if logo_path:
            try:
                self.image(logo_path, x=10, y=8, w=25)
            except:
                pass
        self.set_font("Arial", "B", 12)
        self.cell(0, 5, "University of Central Asia", ln=True, align="C")
        self.cell(0, 5, "310 Lenin Street, Naryn, Kyrgyzstan", ln=True, align="C")
        self.cell(0, 5, "Phone: +996 123456789 | Email: clinic@ucentralasia.org", ln=True, align="C")
        self.ln(10)
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "Leishmaniasis Diagnostic Report", ln=True, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 0, "C")

    def patient_info(self, name, age, gender, dob):
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Patient Name: {name}", ln=True)
        self.cell(0, 10, f"Age: {age}", ln=True)
        self.cell(0, 10, f"Gender: {gender}", ln=True)
        self.cell(0, 10, f"Date of Birth: {dob}", ln=True)
        self.ln(5)

    def add_result(self, image_name, prediction, confidence, actual):
        self.set_font("Arial", "", 12)
        self.cell(0, 10, f"Image: {image_name}", ln=True)
        self.cell(0, 10, f"Prediction: {prediction}", ln=True)
        self.cell(0, 10, f"Confidence: {confidence}", ln=True)
        self.cell(0, 10, f"Actual Label: {actual}", ln=True)
        self.ln(5)

    def signature(self):
        self.set_y(-40)
        self.set_font("Arial", "", 12)
        self.cell(0, 10, "Doctor's Signature: ____________________", ln=True)
        self.ln(10)

    def diagnosis_summary(self, results):
        positive = sum(1 for r in results if r["Prediction"] == "Positive")
        negative = sum(1 for r in results if r["Prediction"] == "Negative")
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"Diagnosis Summary: {positive} Positive, {negative} Negative", ln=True)
        self.ln(10)

    def qr_code(self, patient_name, patient_dob):
        qr = qrcode.make(f"Patient: {patient_name}, DOB: {patient_dob}")
        qr_path = "qr_temp.png"
        qr.save(qr_path)
        self.image(qr_path, x=10, y=self.get_y(), w=30)
        os.remove(qr_path)

    def model_info(self, model_name):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, f"Model Used for Prediction: {model_name}", ln=True)
        self.ln(5)
    def watermark(self, text="UCA Clinic"):
        self.set_font("Arial", "B", 40)
        self.set_text_color(200, 200, 200)
        text_width = self.get_string_width(text)
        x = (self.w - text_width) / 2
        y = self.h / 2
        self.rotate(30, x=x, y=y)
        self.text(x, y, text)
        self.rotate(0)
        self.set_text_color(0, 0, 0)



# ----------------- Prediction Logic -----------------
if uploaded_files:
    results = []
    target_size = (224, 224)

    st.write("### ✅ Optional: Input Actual Labels")
    actual_labels = st.text_area("Enter actual labels (comma-separated):", "")
    actual_labels = [label.strip() for label in actual_labels.split(",") if label.strip()]

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file)

            if selected_model == "Multi-Channel Model":
                prediction, confidence = predict_image_channels(image, model, target_size)
                predicted_class = 1 if prediction == 1 else 0
            else:
                processed_image = preprocess_image(image, target_size)
                predictions = model.predict(processed_image)
                predicted_class = 1 if predictions[0][0] >= 0.5 else 0
                confidence = predictions[0][0] if predicted_class == 1 else 1 - predictions[0][0]

            actual_label = actual_labels[idx] if idx < len(actual_labels) else "Unknown"
            prediction_label = "Positive" if predicted_class == 1 else "Negative"

            results.append({
                "Image Name": uploaded_file.name,
                "Prediction": prediction_label,
                "Confidence": f"{confidence:.2f}",
                "Actual Label": actual_label
            })

            st.image(image, caption=uploaded_file.name, use_column_width=True)
            st.write(f"**Prediction:** {prediction_label}")
            st.write(f"**Confidence:** {confidence:.2f}")
            st.write(f"**Actual Label:** {actual_label}")
            st.write("---")
        except Exception as e:
            st.error(f"Error with {uploaded_file.name}: {e}")

    results_df = pd.DataFrame(results)
    st.write("### 📊 Diagnostic Summary Table")
    st.dataframe(results_df)

    if len(actual_labels) == len(uploaded_files):
        fig = px.bar(
            results_df,
            x="Image Name",
            y="Confidence",
            color="Prediction",
            barmode="group",
            text="Actual Label",
            title="Predicted vs Actual (Confidence)",
        )
        st.plotly_chart(fig)

    pdf = PDF()
    pdf.add_page()
    # ONLY watermark text 'UCA Clinic', not the full confidential text
    pdf.watermark("UCA Clinic-Confidential")
    pdf.model_info(selected_model)
    pdf.patient_info(patient_name, patient_age, patient_gender, patient_dob)

    for row in results:
        pdf.add_result(
            image_name=row["Image Name"],
            prediction=row["Prediction"],
            confidence=row["Confidence"],
            actual=row["Actual Label"]
        )

    pdf.diagnosis_summary(results)
    pdf.qr_code(patient_name, patient_dob)
    pdf.signature()

    pdf_bytes = pdf.output(dest="S").encode("latin1")
    pdf_buffer = io.BytesIO(pdf_bytes)

    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name=f"{patient_name.replace(' ', '_')}_Diagnostic_Report.pdf",
        mime="application/pdf"
    )

    if logo_path and os.path.exists(logo_path):
        os.remove(logo_path)
