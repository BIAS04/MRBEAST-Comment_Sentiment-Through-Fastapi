import uvicorn
import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Imports for creating dummy models if files are missing (for demonstration purposes)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# 1. Initialize the FastAPI App
app = FastAPI(
    title="NLP Model Deployment API",
    description="API to predict MrBeast_Youtube Comment Sentiment",
    version="1.0"
)

# --- CRITICAL FIX: CORS Middleware ---
# This allows your HTML file to communicate with this Python server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# 2. Define filenames
LR_FILENAME = 'lr_model.pkl'
NB_FILENAME = 'nb_model.pkl'
VECTORIZER_FILENAME = 'tfidf_vectorizer.pkl'

# Global variables to hold the models
models = {}

# 3. Define the Input Data Schema
# Updated to match the Frontend request structure
class PredictionRequest(BaseModel):
    text: str
    model_type: str = "lr"  # Default to Logistic Regression

# --- HELPER: Create Dummy Models (Only runs if files are missing) ---
def create_dummy_models_if_missing():
    """Checks if model files exist. If not, creates simple ones."""
    if not os.path.exists(LR_FILENAME) or not os.path.exists(NB_FILENAME):
        print("⚠️ Model files not found. Creating dummy models...")
        
        texts = [
            "I love this product it is amazing",  # Positive
            "This is just okay, nothing special", # Neutral
            "I hate this, it is terrible",        # Negative
            "Excellent work, very happy",         # Positive
            "Average performance",                # Neutral
            "Worst experience ever, very bad"     # Negative
        ]
        # 0 = POSITIVE, 1 = NEUTRAL, 2 = NEGATIVE
        labels = [0, 1, 2, 0, 1, 2] 

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, VECTORIZER_FILENAME)

        lr = LogisticRegression()
        lr.fit(X, labels)
        joblib.dump(lr, LR_FILENAME)

        nb = MultinomialNB()
        nb.fit(X, labels)
        joblib.dump(nb, NB_FILENAME)
        print("✅ Dummy models created successfully.")

# 4. Load Models on Startup
@app.on_event("startup")
def load_models():
    create_dummy_models_if_missing()
    try:
        models['lr'] = joblib.load(LR_FILENAME)
        models['nb'] = joblib.load(NB_FILENAME)
        models['vectorizer'] = joblib.load(VECTORIZER_FILENAME)
        print("✅ All models and vectorizers loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# 5. Main Prediction Endpoint
@app.post("/predict")
def predict_sentiment(request: PredictionRequest):
    if 'vectorizer' not in models:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    # Select Model based on request
    model_key = request.model_type.lower()
    if model_key not in models:
        # Fallback to 'lr' if invalid type sent
        model_key = 'lr'
    
    selected_model = models[model_key]
    vectorizer = models['vectorizer']

    # Transform and Predict
    try:
        features = vectorizer.transform([request.text])
        prediction = selected_model.predict(features)[0]
        
        # Get confidence if available
        probability = 0.0
        if hasattr(selected_model, "predict_proba"):
            probability = float(selected_model.predict_proba(features).max())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map prediction to string label
    # Adjust this map based on your actual training labels!
    # Assuming: 0 = POSITIVE, 1 = NEUTRAL, 2 = NEGATIVE
    sentiment_map = {0: "POSITIVE", 1: "NEUTRAL", 2: "NEGATIVE"}
    
    # If your model predicts strings directly (e.g., "Positive"), use this instead:
    # result_label = str(prediction)
    result_label = sentiment_map.get(int(prediction), str(prediction))

    return {
        "model_used": "Logistic Regression" if model_key == 'lr' else "Naive Bayes",
        "prediction": result_label,
        "confidence": probability
    }

@app.get("/")
def home():
    return {"message": "NLP API is running. Send POST requests to /predict"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)