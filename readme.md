# VITALSCAN ML API (Python/FastAPI)

This is a standalone Python/FastAPI server that serves pre-trained machine learning models for a health diagnosis application.

It is designed to be called by a separate frontend service.

## 🚀 Live API

* **Base URL:** [![Live Demo](https://img.shields.io/badge/Render-Live_Demo-46E3B7?logo=render)](https://vitalscan-api-ml-service.onrender.com)
* **Interactive Docs (Swagger UI):** [![Live Demo](https://img.shields.io/badge/Render-Live_Demo-46E3B7?logo=render)](https://vitalscan-api-ml-service.onrender.com/docs)


> **Note:** This API is deployed on Render's free tier. It will "spin down" after 15 minutes of inactivity and may take 30-60 seconds to "wake up" on the first request.

## Endpoints

### 1. Diabetes Predictor

* **Endpoint:** `POST /predict/diabetes`
* **Model:** `LogisticRegression` (Scikit-learn)
* **Input (JSON):**
    ```json
    {
      "Pregnancies": 6,
      "Glucose": 148.0,
      "BloodPressure": 72.0,
      "SkinThickness": 35.0,
      "Insulin": 0.0,
      "BMI": 33.6,
      "DiabetesPedigreeFunction": 0.627,
      "Age": 50
    }
    ```

### 2. Heart Disease Predictor

* **Endpoint:** `POST /predict/heart_disease`
* **Model:** `LogisticRegression` (Scikit-learn) with data preprocessing.
* **Input (JSON):**
    ```json
    {
      "age": 63,
      "sex": "Male",
      "cp": "typical angina",
      "trestbps": 145.0,
      "chol": 233.0,
      "fbs": "True",
      "restecg": "lv hypertrophy",
      "thalach": 150.0,
      "exang": "False",
      "oldpeak": 2.3,
      "slope": "downsloping",
      "ca": 0.0,
      "thal": "fixed defect"
    }
    ```

### 3. Pneumonia Detector (Image Upload)

* **Endpoint:** `POST /predict/pneumonia`
* **Model:** `Convolutional Neural Network` (TensorFlow/Keras)
* **Input (Form-Data):**
    * **Key:** `file`
    * **Value:** An image file (e.g., `my_xray.jpg`)

---

## Model Storage & Deployment

The trained model files (.pkl, .keras, .tflite) are large and are not stored in this repository's main branch.

Instead, they are hosted as assets in this project's ***GitHub Releases***. The Python ML API is designed to download these model files directly from the release page when the server starts.

The application is deployed on Render as a Python Web Service.

## Project Repositories

* **This API (This Repo):** [https://github.com/Plabss/vitalscan-api.git](https://github.com/Plabss/vitalscan-api.git)
* **Frontend MERN App:** [https://github.com/Plabss/vitalscan-app.git](https://github.com/Plabss/vitalscan-app.git)

## How to Run Locally

1.  Clone this repository.

    ```bash
    git clone https://github.com/Plabss/vitalscan-api.git

    cd vitalscan-api
    ```
2.  Install dependencies (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the FastAPI server:
    ```bash
    uvicorn ml_fastapi:app --reload
    ```
4.  The API will be running at `http://127.0.0.1:8000` and the docs at `http://127.0.0.1:8000/docs`.