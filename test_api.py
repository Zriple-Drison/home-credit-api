"""
Tests unitaires pour l'API Home Credit deployee sur Render
"""

import pytest
import requests
import json

# URL de l'API deployee sur Render
API_URL = "https://home-credit-api.onrender.com"

# ============================================================================
# TEST 1 : L'API repond (Health Check)
# ============================================================================

def test_health_endpoint():
    """Verifie que l'endpoint /health fonctionne"""
    response = requests.get(f"{API_URL}/health")
    
    assert response.status_code == 200, "L'API ne repond pas"
    
    data = response.json()
    assert "status" in data, "Pas de champ 'status'"
    assert "n_features" in data, "Pas de champ 'n_features'"
    assert "optimal_threshold" in data, "Pas de champ 'optimal_threshold'"
    
    assert data["status"] == "healthy", "L'API n'est pas healthy"
    assert data["n_features"] == 85, "Nombre de features incorrect"
    assert data["optimal_threshold"] == 0.35, "Seuil optimal incorrect"
    
    print("✅ Health check OK")


# ============================================================================
# TEST 2 : Prediction client FAIBLE RISQUE -> ACCORDE
# ============================================================================

def test_predict_low_risk_client():
    """Teste la prediction pour un client a faible risque"""
    
    # Donnees d'un client a faible risque (extrait simplifie)
    client_data = {
        "data": {
            "EXT_SOURCE_2": 0.419387842144658,
            "EXT_SOURCE_3": 0.6833170531786634,
            "DAYS_BIRTH": -27858.846009107183,
            "DAYS_REGISTRATION": -2445.7428397662643,
            "DAYS_EMPLOYED": -2402.736317563084,
            "AMT_CREDIT": 151705.66400065727,
            "DAYS_ID_PUBLISH": -4991.882317039751,
            "AMT_ANNUITY": 10671.388291809786,
            "BUREAU_DAYS_CREDIT_MAX_x": -1255.8601325635364,
            "delay_mean": -9.396752340908987,
            "PREV_DAYS_DECISION_MAX": -675.0818344226923,
            "EXT_SOURCE_1": 0.5031474047380476,
            "BUREAU_AMT_CREDIT_SUM_MEAN_x": 115134.7805511389,
            "PREV_CREDIT_APPLICATION_RATIO_MEAN": 0.6405405423615701,
            "DAYS_LAST_PHONE_CHANGE": 0.0,
            "BUREAU_AMT_CREDIT_SUM_SUM_x": 209704.83787909712,
            "instalment_mean": 15234.038914554534,
            "BUREAU_DAYS_CREDIT_MIN_x": -2396.483779939765,
            "PREV_AMT_APPLICATION_MEAN": 242525.75556302548,
            "POS_CNT_INSTALMENT_MEAN": 11.726874369555912,
            "PREV_DAYS_DECISION_MEAN": -1472.7464721596998,
            "BUREAU_DAYS_CREDIT_MEAN_x": -1166.3771611006136,
            "PREV_AMT_ANNUITY_MEAN": 7730.18953960453,
            "PREV_DAYS_DECISION_MIN": -2344.3736714265533,
            "PREV_AMT_ANNUITY_MAX": 24240.656614756677,
            "LOG_INCOME": 13.866980516125738,
            "instalment_sum": 705223.2237017488,
            "PREV_AMT_APPLICATION_MAX": 381966.2195060758,
            "nb_instalments": 65.19396069579012,
            "POS_MONTHS_BALANCE_COUNT": 120.21124658128764,
            "PREV_AMT_DOWN_PAYMENT_MEAN": 14164.222973065147,
            "delay_rate": 0.011039128392619677,
            "PREV_AMT_DOWN_PAYMENT_MAX": 62758.02045637293,
            "BUREAU_CREDIT_ACTIVE_ACTIVE_MEAN_x": 0.0,
            "HOUR_APPR_PROCESS_START": 12.77512897548386,
            "partial_payment_rate": 0.0,
            "LANDAREA_AVG": 0.03618672278386984,
            "POS_MONTHS_BALANCE_MIN": -135.8678342764103,
            "OWN_CAR_AGE": 6.891170083796371,
            "LIVINGAREA_AVG": 0.10398827474673883,
            "BUREAU_BB_STATUS0_RATE_MEAN_x": 0.26337023805410575,
            "APARTMENTS_AVG": 0.10131489296459367,
            "BASEMENTAREA_AVG": 0.10735299876357693,
            "NONLIVINGAREA_AVG": 0.002407497936878282,
            "YEARS_BUILD_AVG": 1.1088805619960675,
            "PREV_IS_APPROVED_MEAN": 0.9528833604155117,
            "COMMONAREA_AVG": 0.030504588133203468,
            "PREV_IS_REFUSED_MEAN": 0.0,
            "BUREAU_BB_NB_MONTHS_MEAN_x": 29.12629607984366,
            "POS_CNT_INSTALMENT_MAX": 19.23808487696524,
            "LIVINGAPARTMENTS_AVG": 0.05203566966936283,
            "PREV_SK_ID_PREV_COUNT": 4.018675231125497,
            "POS_POS_STATUS_SIGNED_MEAN": 0.0,
            "CC_CNT_DRAWINGS_CURRENT_MEAN": 0.0,
            "CC_MONTHS_BALANCE_COUNT": 0.0,
            "CC_CC_UTILIZATION_MAX": 0.0,
            "BUREAU_AMT_CREDIT_SUM_LIMIT_MEAN_x": 0.0,
            "CC_AMT_BALANCE_MEAN": 0.0,
            "BUREAU_AMT_CREDIT_SUM_LIMIT_MAX_x": 0.0,
            "ENTRANCES_AVG": 0.08177709768337733,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 0.6438679663433298,
            "CC_CC_UTILIZATION_MEAN": 0.0,
            "BUREAU_BB_STATUS0_RATE_MEAN_y": 0.0,
            "BUREAU_BB_NB_MONTHS_MEAN_y": 0.0,
            "FLOORSMAX_AVG": 0.10079689875032337,
            "CC_CNT_DRAWINGS_CURRENT_MAX": 0.0,
            "CNT_CHILDREN": 0.0,
            "REGION_RATING_CLIENT": 1.8458870874831155,
            "FLOORSMIN_AVG": 0.26986896860820825,
            "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
            "FLAG_WORK_PHONE": 0.0,
            "ELEVATORS_AVG": 0.0,
            "CC_CC_STATUS_ACTIVE_MEAN": 0.0,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
            "FLAG_DOCUMENT_3": 0.6413835542163402,
            "FLAG_PHONE": 0.0,
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM_x": 0.0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
            "REG_CITY_NOT_WORK_CITY": 0.0,
            "LIVE_CITY_NOT_WORK_CITY": 0.0,
            "REG_CITY_NOT_LIVE_CITY": 0.0,
            "FLAG_EMP_PHONE": 0.0,
            "FLAG_DOCUMENT_6": 0.0,
            "FLAG_DOCUMENT_16": 0.0,
            "FLAG_DOCUMENT_13": 0.0
        }
    }
    
    response = requests.post(f"{API_URL}/predict", json=client_data)
    
    assert response.status_code == 200, "Erreur API"
    
    data = response.json()
    assert "probability" in data, "Pas de probabilite"
    assert "decision" in data, "Pas de decision"
    
    assert data["probability"] < 0.35, "Probabilite trop elevee pour un client faible risque"
    assert data["decision"] == "ACCORDÉ", "Decision incorrecte"
    assert data["risk_level"] == "FAIBLE", "Niveau de risque incorrect"
    
    print(f"✅ Client faible risque : {data['probability']:.1%} -> {data['decision']}")


# ============================================================================
# TEST 3 : Donnees INVALIDES -> Erreur 422
# ============================================================================

def test_predict_invalid_data():
    """Teste que l'API rejette les donnees invalides"""
    
    # Donnees invalides (manque le champ 'data')
    invalid_data = {
        "wrong_field": {"EXT_SOURCE_2": 0.5}
    }
    
    response = requests.post(f"{API_URL}/predict", json=invalid_data)
    
    assert response.status_code == 422, "L'API devrait rejeter les donnees invalides"
    
    print("✅ Donnees invalides rejetees correctement")


# ============================================================================
# TEST 4 : Info modele
# ============================================================================

def test_model_info_endpoint():
    """Verifie que l'endpoint /model-info fonctionne"""
    response = requests.get(f"{API_URL}/model-info")
    
    assert response.status_code == 200, "L'API ne repond pas"
    
    data = response.json()
    assert "model_type" in data, "Pas de type de modele"
    assert "auc_test" in data, "Pas d'AUC"
    
    assert data["model_type"] == "LightGBM", "Type de modele incorrect"
    assert data["auc_test"] > 0.75, "AUC trop faible"
    
    print(f"✅ Model info OK - AUC: {data['auc_test']}")


# ============================================================================
# EXECUTION DES TESTS
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TESTS DE L'API HOME CREDIT (RENDER)")
    print("="*70 + "\n")
    
    print("⚠️  Attention : Le premier test peut prendre 30-60 secondes")
    print("    (l'API Render se reveille du mode veille)\n")
    
    # Executer les tests
    pytest.main([__file__, "-v", "-s"])
