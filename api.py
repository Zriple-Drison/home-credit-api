"""
API FastAPI pour le modèle Home Credit
Prédiction de la probabilité de défaut de paiement
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List

# ============================================================================
# INITIALISATION DE L'API
# ============================================================================

app = FastAPI(
    title="Home Credit API",
    description="API de prédiction de risque de crédit",
    version="1.0.0"
)

# ============================================================================
# CHARGEMENT DU MODÈLE ET DES MÉTADONNÉES
# ============================================================================

print("🚀 Chargement du modèle...")

try:
    # Charger le modèle
    model = joblib.load("best_model_lgbm.pkl")
    print("✅ Modèle chargé")
    
    # Charger les features
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print(f"✅ {len(feature_names)} features chargées")
    
    # Charger les infos du modèle
    with open("model_info.json", "r") as f:
        model_info = json.load(f)
    print(f"✅ Métadonnées chargées")
    
    OPTIMAL_THRESHOLD = model_info["optimal_threshold"]
    print(f"✅ Seuil optimal : {OPTIMAL_THRESHOLD}")
    
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")
    raise

# ============================================================================
# MODÈLES DE DONNÉES (PYDANTIC)
# ============================================================================

class PredictionRequest(BaseModel):
    """Requête de prédiction - Données du client"""
    data: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "data": {
                    "AMT_CREDIT": 450000.0,
                    "AMT_INCOME_TOTAL": 180000.0,
                    "AGE_YEARS": 35.0,
                    # ... autres features
                }
            }
        }

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    probability: float
    decision: str
    threshold: float
    risk_level: str
    message: str

class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    model_loaded: bool
    n_features: int
    optimal_threshold: float

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=dict)
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "API Home Credit - Prédiction de risque de crédit",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Vérifier l'état de l'API",
            "/predict": "Faire une prédiction",
            "/model-info": "Informations sur le modèle"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifier que l'API fonctionne"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        n_features=len(feature_names),
        optimal_threshold=OPTIMAL_THRESHOLD
    )

@app.get("/model-info", response_model=dict)
async def get_model_info():
    """Obtenir les informations sur le modèle"""
    return model_info

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Prédire la probabilité de défaut pour un client
    
    Args:
        request: Données du client (dictionnaire de features)
    
    Returns:
        Probabilité, décision, et niveau de risque
    """
    try:
        # 1. Convertir les données en DataFrame
        df = pd.DataFrame([request.data])
        
        # 2. Vérifier que toutes les features sont présentes
        missing_features = set(feature_names) - set(df.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Features manquantes : {list(missing_features)[:10]}... ({len(missing_features)} au total)"
            )
        
        # 3. Réordonner les colonnes dans le bon ordre
        df = df[feature_names]
        
        # 4. Faire la prédiction
        proba = model.predict_proba(df)[0, 1]  # Probabilité de la classe 1 (défaut)
        
        # 5. Décision selon le seuil optimal
        decision = "REFUSÉ" if proba >= OPTIMAL_THRESHOLD else "ACCORDÉ"
        
        # 6. Niveau de risque
        if proba < 0.2:
            risk_level = "FAIBLE"
        elif proba < 0.4:
            risk_level = "MODÉRÉ"
        elif proba < 0.6:
            risk_level = "ÉLEVÉ"
        else:
            risk_level = "TRÈS ÉLEVÉ"
        
        # 7. Message personnalisé
        if decision == "ACCORDÉ":
            message = f"Crédit accordé. Risque de défaut : {proba:.1%}"
        else:
            message = f"Crédit refusé. Risque de défaut trop élevé : {proba:.1%}"
        
        return PredictionResponse(
            probability=float(proba),
            decision=decision,
            threshold=OPTIMAL_THRESHOLD,
            risk_level=risk_level,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )

# ============================================================================
# LANCEMENT DE L'API
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🚀 LANCEMENT DE L'API HOME CREDIT")
    print("="*70)
    print("📍 URL : http://localhost:8000")
    print("📖 Documentation : http://localhost:8000/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
