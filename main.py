"""
API de Predição de Crédito - Versão Simplificada
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="6 características numéricas")
    model_name: Optional[str] = Field(default="logistic_regression")
    
    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if len(v) != 6:
            raise ValueError("São necessárias 6 características")
        return v

class PredictionResponse(BaseModel):
    prediction: float
    probability: Optional[float] = None
    confidence: str
    model_used: str
    recommendation: str

class ModelManager:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.model_names = {
            "logistic_regression": "Regressão Logística",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting"
        }
        self.load_models()
    
    def load_models(self):
        logger.info("Carregando modelos...")
        
        
        if Path("scaler.joblib").exists():
            try:
                self.scaler = joblib.load("scaler.joblib")
                logger.info("Scaler carregado")
            except Exception as e:
                logger.warning(f"Erro scaler: {e}")
        
        for model_key in self.model_names.keys():
            model_path = f"{model_key}.joblib"
            if Path(model_path).exists():
                try:
                    self.models[model_key] = joblib.load(model_path)
                    logger.info(f"Modelo {model_key} carregado")
                except Exception as e:
                    logger.error(f"Erro {model_key}: {e}")
        
        logger.info(f"{len(self.models)} modelos disponíveis")
    
    def predict(self, features: List[float], model_name: str = "logistic_regression"):
        if model_name not in self.models:
            available = list(self.models.keys())
            if not available:
                raise HTTPException(status_code=503, detail="Nenhum modelo disponível")
            model_name = available[0]
        
        try:
            X = np.array(features).reshape(1, -1)
            if self.scaler:
                X = self.scaler.transform(X)
            
            model = self.models[model_name]
            prediction = float(model.predict(X)[0])
            
            probability = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)[0]
                probability = float(max(probs))
            
            score = probability if probability else prediction
            confidence = "Alto" if score >= 0.8 else "Médio" if score >= 0.6 else "Baixo"
            
            if prediction >= 0.7:
                recommendation = "Aprovação recomendada"
            elif prediction >= 0.4:
                recommendation = "Analisar com cuidado"
            else:
                recommendation = "Não recomendado"
            
            return {
                "prediction": prediction,
                "probability": probability,
                "confidence": confidence,
                "model_used": self.model_names[model_name],
                "recommendation": recommendation
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erro: {str(e)}")


model_manager = ModelManager()

app = FastAPI(
    title="API de Predição de Crédito",
    version="2.0.0",
    description="API para predição de aprovação de crédito usando ML",
    docs_url="/",
    redoc_url="/docs"
)

@app.get("/health")
async def health():
    return {
        "message": "API funcionando",
        "models_loaded": len(model_manager.models),
        "available_models": list(model_manager.models.keys())
    }

@app.get("/models")
async def list_models():
    return {
        "models": [
            {"key": key, "name": model_manager.model_names[key]}
            for key in model_manager.models.keys()
        ]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    result = model_manager.predict(request.features, request.model_name)
    return PredictionResponse(**result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    print("🏦 API de Predição de Crédito")
    print(f"Servidor: http://{host}:{port}")
    
    uvicorn.run("main:app", host=host, port=port, reload=True)
