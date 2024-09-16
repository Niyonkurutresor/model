from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Step 1: Load the saved model
model = joblib.load('disease_prediction_model.pkl')

# Step 2: Create a FastAPI app
app = FastAPI()

# Step 3: Define a request body model
class Symptoms(BaseModel):
    symptoms: str

# Step 4: Define the prediction route
@app.post("/predict_disease/")
async def predict_disease(data: Symptoms):
    symptoms = data.symptoms
    prediction = model.predict([symptoms])
    return {"predicted_disease": prediction[0]}

# Step 5: Run FastAPI using Uvicorn
# In the terminal run: uvicorn script_name:app --reload
