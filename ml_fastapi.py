import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import io
from fastapi import File, UploadFile

import os
import requests

# Initialize the FastAPI app
app = FastAPI(title="VitalScan AI Diagnosis API", version="1.0")

# --- Model Configuration ---
MODEL_URLS = {
    "diabetes": "https://github.com/Plabss/vitalscan-api/releases/download/v1.0-models/diabetes_model.pkl",
    "heart_model": "https://github.com/Plabss/vitalscan-api/releases/download/v1.0-models/heart_disease_model.pkl",
    "heart_columns": "https://github.com/Plabss/vitalscan-api/releases/download/v1.0-models/heart_disease_model_columns.pkl",
    "heart_scaler": "https://github.com/Plabss/vitalscan-api/releases/download/v1.0-models/heart_disease_scaler.pkl",
    "pneumonia": "https://github.com/Plabss/vitalscan-api/releases/download/v1.0-models/pneumonia_model.tflite"
}

# Use /tmp for ephemeral filesystem storage
MODEL_DIR = "/tmp/models"

# Global variables to hold loaded models
model = None
heart_model = None
heart_scaler = None
heart_model_columns = None
pneumonia_model = None


def download_file(url, local_path):
    """Downloads a file from a URL, skipping if it already exists."""
    if os.path.exists(local_path):
        print(f"File already exists: {local_path}")
        return
    
    print(f"Downloading {url} to {local_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            # Ensure directory exists
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        raise


@app.on_event("startup")
async def load_models():
    """On app startup, download and load all models into global variables."""
    global model, heart_model, heart_scaler, heart_model_columns, pneumonia_model
    
    os.makedirs(MODEL_DIR, exist_ok=True)
        
    try:
        # Diabetes
        db_model_path = os.path.join(MODEL_DIR, "diabetes_model.pkl")
        download_file(MODEL_URLS["diabetes"], db_model_path)
        model = joblib.load(db_model_path)
        print("Diabetes model loaded.")
        
        # Heart Disease
        hd_model_path = os.path.join(MODEL_DIR, "heart_disease_model.pkl")
        download_file(MODEL_URLS["heart_model"], hd_model_path)
        heart_model = joblib.load(hd_model_path)
        print("Heart disease model loaded.")

        hd_cols_path = os.path.join(MODEL_DIR, "heart_disease_model_columns.pkl")
        download_file(MODEL_URLS["heart_columns"], hd_cols_path)
        heart_model_columns = joblib.load(hd_cols_path)
        print("Heart disease columns loaded.")

        hd_scaler_path = os.path.join(MODEL_DIR, "heart_disease_scaler.pkl")
        download_file(MODEL_URLS["heart_scaler"], hd_scaler_path)
        heart_scaler = joblib.load(hd_scaler_path)
        print("Heart disease scaler loaded.")
        
        # Pneumonia (TFLite)
        pneu_model_path = os.path.join(MODEL_DIR, "pneumonia_model.tflite")
        download_file(MODEL_URLS["pneumonia"], pneu_model_path)
        
        # Load TFLite model and allocate tensors
        pneumonia_model = tf.lite.Interpreter(model_path=pneu_model_path)
        pneumonia_model.allocate_tensors()
        
        print("Pneumonia TFLite model loaded and tensors allocated.")
        
        print("All models loaded successfully.")
        
    except Exception as e:
        print(f"Error loading models: {e}")


# --- Pydantic Input Models ---

class DiabetesFeatures(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int
    class Config:
        json_schema_extra = {"example": {"Pregnancies": 6, "Glucose": 148.0, "BloodPressure": 72.0, "SkinThickness": 35.0, "Insulin": 0.0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.627, "Age": 50}}

class HeartDiseaseFeatures(BaseModel):
    age: int
    sex: str 
    cp: str  
    trestbps: float
    chol: float
    fbs: str   
    restecg: str 
    thalach: float
    exang: str 
    oldpeak: float
    slope: str 
    ca: float
    thal: str 
    class Config:
        json_schema_extra = {"example": {"age": 63, "sex": "Male", "cp": "typical angina", "trestbps": 145.0, "chol": 233.0, "fbs": "True", "restecg": "lv hypertrophy", "thalach": 150.0, "exang": "False", "oldpeak": 2.3, "slope": "downsloping", "ca": 0.0, "thal": "fixed defect"}}


# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Healthcare Diagnosis API!"}


@app.post("/predict/diabetes")
async def predict_diabetes(features: DiabetesFeatures):
    if model is None:
        return {"error": "Model not loaded. Please check server logs."}
    
    data = np.array([[features.Pregnancies, features.Glucose, features.BloodPressure, features.SkinThickness, features.Insulin, features.BMI, features.DiabetesPedigreeFunction, features.Age]])
    
    prediction_raw = model.predict(data)
    prediction_proba = model.predict_proba(data)
    
    prediction = int(prediction_raw[0])
    confidence = float(prediction_proba[0][prediction])
    diagnosis = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    return {"diagnosis": diagnosis, "prediction_value": prediction, "confidence_score": round(confidence, 4)}


@app.post("/predict/heart_disease")
async def predict_heart_disease(features: HeartDiseaseFeatures):
    if not all([heart_model, heart_scaler, heart_model_columns]):
        return {"error": "Heart disease model is not loaded. Check server logs."}
    
    # Process input features to match model's training data
    input_data = pd.DataFrame([features.dict()])
    input_data_encoded = pd.get_dummies(input_data)
    # Ensure all columns from training are present, filling missing with 0
    input_data_reindexed = input_data_encoded.reindex(columns=heart_model_columns, fill_value=0)
    # Scale data
    input_data_scaled = heart_scaler.transform(input_data_reindexed)
    
    prediction_raw = heart_model.predict(input_data_scaled)
    prediction_proba = heart_model.predict_proba(input_data_scaled)
    
    prediction = int(prediction_raw[0])
    confidence = float(prediction_proba[0][prediction])
    diagnosis = "Heart Disease Positive" if prediction == 1 else "Heart Disease Negative"
    
    return {"diagnosis": diagnosis, "prediction_value": prediction, "confidence_score": round(confidence, 4)}


@app.post("/predict/pneumonia")
async def predict_pneumonia(file: UploadFile = File(...)):
    if not pneumonia_model:
        return {"error": "Pneumonia model is not loaded. Check server logs."}

    try:
        # Read and preprocess the image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((150, 150))
        image_array = img_to_array(image)
        image_array = image_array / 255.0 # Normalize
        image_array = image_array.astype(np.float32) # TFLite requires float32
        image_array = tf.expand_dims(image_array, 0) # Create batch dimension

        # Run TFLite inference
        input_details = pneumonia_model.get_input_details()
        output_details = pneumonia_model.get_output_details()

        pneumonia_model.set_tensor(input_details[0]['index'], image_array)
        pneumonia_model.invoke()
        prediction_raw = pneumonia_model.get_tensor(output_details[0]['index'])
        
        confidence = float(prediction_raw[0][0])
        
        # Interpret result
        if confidence > 0.5:
            diagnosis = "Pneumonia"
            confidence_score = confidence
        else:
            diagnosis = "Normal"
            confidence_score = 1.0 - confidence

        return {
            "diagnosis": diagnosis,
            "confidence_score": round(confidence_score, 4)
        }

    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}
