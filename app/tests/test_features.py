from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_features():
    response = client.get("/features")
    assert response.status_code == 200
    assert set(response.json()) == {
                                    "heure_supplementaires",
                                    "age",
                                    "FE_ratio_ancienneté",
                                    "FE_cadre",
                                    "frequence_deplacement",
                                    "FE_duree_moy_exp_precedentes",
                                    "FE_ratio_evolution",
                                    "niveau_education",
                                    "FE_reste_plus_longtemps",
                                    "poste",
                                    "statut_marital"
                                    }
