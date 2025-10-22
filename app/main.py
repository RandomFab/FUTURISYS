# app/main.py
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel
from typing import Literal
import pandas as pd

app = FastAPI()

class PredictionData(BaseModel):
  heure_supplementaires: Literal[0,1]
  age: int
  FE_ratio_ancienneté:float
  FE_cadre: Literal[0,1]
  frequence_deplacement: Literal[1,2,3]
  FE_duree_moy_exp_precedentes:float
  FE_ratio_evolution:float
  niveau_education: Literal[1,2,3,4,5]
  FE_reste_plus_longtemps: Literal[0,1]
  poste: Literal['Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead']
  statut_marital:Literal['Célibataire','Marié(e)','Divorcé(e)']

bundle = load("app/model/model_HR_prediction_TECHNOVA.joblib")
HR_model = bundle['model']
HR_threshold = bundle['threshold']


@app.get('/threshold')
def get_threshold():
    return HR_threshold

@app.get("/features")
def get_features():
    preprocessor = HR_model.named_steps['preprocessing']

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    all_features = list(num_features) + list(cat_features)

    return all_features

@app.get('/model-info')
def get_model_info():
    model = HR_model.named_steps['model']
    infos_model = {
        'type' : model.__class__.__name__,
        'params' : model.get_params(),
        'feature_names' : get_features(),
        'threshold_F1_optimised' : get_threshold()
    }
    return infos_model

@app.post('/predict')
def post_prediction(data: PredictionData):

    """
    Prédit le départ ou non d'un employé

    ARGS : un dictionnaire contenant les valeurs des différentes variables 
    {
        "heure_supplementaires": 2,
        "age": 0,
        "FE_ratio_ancienneté": 0,
        "FE_cadre": 0,
        "frequence_deplacement": "régulier",
        "FE_duree_moy_exp_precedentes": 0,
        "FE_ratio_evolution": 0,
        "niveau_education": 1,
        "FE_reste_plus_longtemps": 0,
        "poste": "Assistant de Direction",
        "statut_marital": "Célibataire"
    }

    RETURNS : La probabilité identifié par le modèle et la prédiction en fonction du seuil optimisé
    """
        
    df = pd.DataFrame([data.dict()])
    proba = HR_model.predict_proba(df)[0][1]
    predict = (proba > HR_threshold)
    return {'probabilité': round(float(proba),3),
            'prédiction': bool(predict)}



