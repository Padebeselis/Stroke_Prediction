from pydantic import BaseModel
from fastapi import FastAPI
import pandas as pd
import joblib
#from sklearn.naive_bayes import GaussianNB

# Expected input
class Stroke_Event(BaseModel):
    Age: float
    Hypertension: int
    Heart_disease: int
    Avg_glucose_level: float
    Bmi: float 
    Gender_Female: int
    Ever_married: int
    Residence_Urban: int
    Smoking_status_was_missing: int
    Bmi_was_missing: int
    Work_type_Govt_job: int
    Work_type_Private: int
    Work_type_Self_employed: int
    Smoking_status_formerly_smoked: int
    Smoking_status_never_smoked: int
    Smoking_status_smokes: int


# Expected output
class PredictionOut(BaseModel):
    default_proba: float


#model = GaussianNB(var_smoothing=1e-09)
model = joblib.load("model.pkl")

app = FastAPI()

# Home page
@app.get("/")
def home():
    return {"message": "Stroke Prediction App", "model_version": 0.1}



# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: Stroke_Event):
    cust_df = pd.DataFrame([payload.model_dump()])
    predictions = model.predict_proba(cust_df)[0, 1]

    adjusted_predictions = (predictions > 0.545455).astype(int)

    result = {"default_proba": adjusted_predictions}
    return result