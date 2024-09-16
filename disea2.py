from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model
with open('disease_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the structure of the input data using Pydantic
class SymptomsInput(BaseModel):
    # Include all the symptoms your model expects as fields
    symptom1: float
    symptom2: float
    symptom3: float
    symptom4: float
    # Add more fields as needed based on your dataset

# Define a route for making predictions
@app.post("/predict/")
async def predict(symptoms: SymptomsInput):
    # Convert the input data into a DataFrame
    input_data = pd.DataFrame([symptoms.dict()])

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)

    # Return the prediction result
    return {"prediction": prediction[0]}
