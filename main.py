from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load the trained model
with open("number_classifier.pkl", "rb") as f:
    model = pickle.load(f)


@app.get("/")
def read_root():
    return {"message": "Welcome to the number classification API"}

@app.post("/predict/")
def predict_number(number: int):
    # Predict using the trained model
    prediction = model.predict(np.array([[number]]))
    
    # Convert prediction to human-readable label
    labels = {0: 'small', 1: 'medium', 2: 'large'}
    result = labels[int(prediction[0])]
    
    return {"number": number, "prediction": result}
