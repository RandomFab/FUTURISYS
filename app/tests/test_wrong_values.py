from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_wrong_value_heure_sup():
    response = client.post('/predict_from_raw_data',json={
        "heure_supplementaires": 2,
        "age": 35,
        "frequence_deplacement": 0,
        "niveau_education": 1,
        "poste": "Assistant de Direction",
        "statut_marital": "Célibataire",
        "annees_dans_l_entreprise": 0,
        "nombre_experiences_precedentes": 0,
        "annees_dans_le_poste_actuel": 0,
        "annee_experience_totale": 0
    })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail[0]["type"] == 'literal_error'
    assert detail[0]['loc'][1] == 'heure_supplementaires'

def test_wrong_value_frequence_dep():
    response = client.post('/predict_from_raw_data',json={
        "heure_supplementaires": 1,
        "age": 35,
        "frequence_deplacement": 3,
        "niveau_education": 1,
        "poste": "Assistant de Direction",
        "statut_marital": "Célibataire",
        "annees_dans_l_entreprise": 0,
        "nombre_experiences_precedentes": 0,
        "annees_dans_le_poste_actuel": 0,
        "annee_experience_totale": 0
    })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail[0]["type"] == 'literal_error'
    assert detail[0]['loc'][1] == 'frequence_deplacement'

def test_wrong_value_niveau_educ():
    response = client.post('/predict_from_raw_data',json={
        "heure_supplementaires": 1,
        "age": 35,
        "frequence_deplacement": 2,
        "niveau_education": 6,
        "poste": "Assistant de Direction",
        "statut_marital": "Célibataire",
        "annees_dans_l_entreprise": 0,
        "nombre_experiences_precedentes": 0,
        "annees_dans_le_poste_actuel": 0,
        "annee_experience_totale": 0
    })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail[0]["type"] == 'literal_error'
    assert detail[0]['loc'][1] == 'niveau_education'
    
def test_wrong_value_poste():
    response = client.post('/predict_from_raw_data',json={
        "heure_supplementaires": 1,
        "age": 35,
        "frequence_deplacement": 2,
        "niveau_education": 5,
        "poste": "développeur",
        "statut_marital": "Célibataire",
        "annees_dans_l_entreprise": 0,
        "nombre_experiences_precedentes": 0,
        "annees_dans_le_poste_actuel": 0,
        "annee_experience_totale": 0
    })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail[0]["type"] == 'literal_error'
    assert detail[0]['loc'][1] == 'poste'

def test_wrong_value_statut_marital():
    response = client.post('/predict_from_raw_data',json={
        "heure_supplementaires": 1,
        "age": 35,
        "frequence_deplacement": 2,
        "niveau_education": 5,
        "poste": "Assistant de Direction",
        "statut_marital": "Celibataire",
        "annees_dans_l_entreprise": 0,
        "nombre_experiences_precedentes": 0,
        "annees_dans_le_poste_actuel": 0,
        "annee_experience_totale": 0
    })
    assert response.status_code == 422
    detail = response.json()['detail']
    assert detail[0]["type"] == 'literal_error'
    assert detail[0]['loc'][1] == 'statut_marital'