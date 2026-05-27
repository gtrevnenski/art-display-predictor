"""
MET Art Display Predictor API
FastAPI application for Google Cloud Run
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="MET Art Display Predictor",
    description="Predict whether an artwork is likely to be displayed at the Metropolitan Museum of Art",
    version="1.0.0"
)

# Configure CORS for your website
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:1313",
        "https://georgi-trevnenski.dev",  # Update with your actual domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (loaded once at startup)
embedder = None
model = None


class PredictionRequest(BaseModel):
    """Input schema for prediction request"""
    text: str = Field(..., description="Combined text description of the artwork")
    objectEndDate: Optional[int] = Field(None, description="Object creation end date")
    department: Optional[str] = Field(None, description="Department")
    country: Optional[str] = Field(None, description="Country of origin")
    cat1: Optional[str] = Field(None, description="Primary category")
    subcat1: Optional[str] = Field(None, description="Subcategory (classification)")
    cat2: Optional[str] = Field(None, description="Secondary category (object type)")
    
    # Ignore has_country if sent by client (we calculate it)
    class Config:
        extra = "ignore"
        schema_extra = {
            "example": {
                "text": "Oil painting of a landscape from 19th century France",
                "objectEndDate": 1860,
                "department": "European Paintings",
                "country": "France",
                "cat1": "Paintings",
                "subcat1": "Paintings",
                "cat2": "Paintings"
            }
        }


class PredictionResponse(BaseModel):
    """Output schema for prediction response"""
    probability: float = Field(..., description="Probability of being on view (0-1)")
    prediction: str = Field(..., description="'on-view' or 'not-on-view'")
    confidence: str = Field(..., description="'low', 'medium', or 'high'")
    explanation: str = Field(..., description="Human-readable explanation")


@app.on_event("startup")
async def load_models():
    """Load models at startup (runs once when container starts)"""
    global embedder, model
    
    logger.info("Loading models...")
    
    try:
        # Load embedding model (384 dimensions)
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("✓ Embedding model loaded")
        
        # Load CatBoost model
        model = CatBoostClassifier()
        model.load_model("catboost_model__MiniLM-L6-v2.cbm")
        logger.info("✓ CatBoost model loaded")
        
        logger.info("All models loaded successfully!")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MET Art Display Predictor",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "embedder_loaded": embedder is not None,
        "model_loaded": model is not None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for an artwork.
    
    Returns the probability that the artwork would be on display,
    along with a binary prediction and confidence level.
    """
    try:
        # Validate models are loaded
        if embedder is None or model is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        
        # Generate text embedding
        logger.info(f"Generating embedding for text: {request.text[:50]}...")
        embedding = embedder.encode([request.text])[0]
        
        # Prepare features (must match training pipeline exactly)
        # Order: [objectEndDate, has_country, department, country, cat1, subcat1, cat2, emb_0...emb_383]
        features = []
        
        # Add numeric feature (use median if not provided)
        features.append(request.objectEndDate if request.objectEndDate is not None else 1850)
        
        # Add has_country (boolean: 1 if country provided, 0 otherwise)
        features.append(1 if request.country else 0)
        
        # Add categorical features (use "None" string for missing values to match training)
        features.append(request.department if request.department else "None")
        features.append(request.country if request.country else "None")
        features.append(request.cat1 if request.cat1 else "None")
        features.append(request.subcat1 if request.subcat1 else "None")
        features.append(request.cat2 if request.cat2 else "None")
        
        # Add embeddings
        features.extend(embedding)
        
        # Convert to numpy array (object dtype to preserve categorical strings)
        features_array = np.array([features], dtype=object)
        
        logger.info(f"Feature vector shape: {features_array.shape}")
        
        # Make prediction with Pool to specify categorical features
        from catboost import Pool
        feature_names = ['objectEndDate', 'has_country', 'department', 'country', 'cat1', 'subcat1', 'cat2'] + [f'emb_{i}' for i in range(384)]
        cat_features = [2, 3, 4, 5, 6]  # Indices of categorical features: department, country, cat1, subcat1, cat2
        
        pred_pool = Pool(features_array, feature_names=feature_names, cat_features=cat_features)
        proba = model.predict_proba(pred_pool)[0][1]
        
        # Use optimal threshold from training (0.722)
        prediction = "on-view" if proba > 0.722 else "not-on-view"
        
        # Determine confidence level
        if proba > 0.85 or proba < 0.15:
            confidence = "high"
        elif proba > 0.7 or proba < 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate explanation
        if prediction == "on-view":
            explanation = f"This artwork has a {proba*100:.1f}% probability of being on display. The model suggests it would likely be exhibited."
        else:
            explanation = f"This artwork has a {(1-proba)*100:.1f}% probability of not being on display. It's more likely to be put in storage."
        
        logger.info(f"Prediction: {prediction} ({proba:.3f})")
        
        return PredictionResponse(
            probability=float(proba),
            prediction=prediction,
            confidence=confidence,
            explanation=explanation
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(requests: list[PredictionRequest]):
    """
    Make predictions for multiple artworks at once.
    Useful for bulk processing or testing.
    """
    if len(requests) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 items per batch")
    
    results = []
    for req in requests:
        try:
            result = await predict(req)
            results.append(result.dict())
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

