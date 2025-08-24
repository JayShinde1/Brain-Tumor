from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated
import pickle
import pandas as pd
from PIL import Image
import numpy as np
import os, io, json

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(title="Brain Tumor Detection API")

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

class PatientInfo(BaseModel):
    name: Annotated[str, Field(..., min_length=1, max_length=100, description='Name of the patient')]
    age: Annotated[int, Field(..., gt = 0, lt = 120, description = 'Age of the user.')]
    sex: Annotated[Literal['Male','Female','male','female'], Field(description='Sex of the user, Male or Female')]

    @field_validator('sex')
    def validate_sex(cls, v):
        if v.lower() not in ['male','female']:
            raise ValueError("Sex must be Male or Female")
        return v.title()  
    
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.resize((256,256)).convert("RGB")
    img_arr = np.array(image)/255.0
    return np.expand_dims(img_arr, axis = 0)

def validate_image(file: UploadFile) -> bool:
    img_extentions = ['.jpg','.jpeg','.png','.bmp']
    file_extention = os.path.splitext(file.filename.lower())[1]
    return file_extention in img_extentions

@app.get("/")
def root():
    return {
        "message":"Brain Tumor Detection API.",
        "model_path_exists": os.path.exists("model/model.pkl")  
    }

@app.post("/predict/")
async def tumor_predict_karo(
    file: UploadFile = File(...),
    name: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...)
):
    try:

        patient = PatientInfo(name=name, age=age, sex=sex)

        if not validate_image(file):
            raise HTTPException(status_code=404, detail = "Invalid image format")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess(image)

        prediction = model.predict(input_tensor, verbose = 1)
        result = int(np.argmax(prediction))
        confidence = float(prediction[0][result]) * 100 

        all_predictions = {
            labels[i]: round(float(prediction[0][i]) * 100, 2)
            for i in range(len(labels))
        }

        return {
            "Prediction": labels[result],
            "Accuracy": f"{confidence:.2f}%",
            "All predictions": all_predictions,
            "Patient info.": patient.model_dump(),
            
        }
    
        
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))