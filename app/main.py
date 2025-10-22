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
  FE_cadre: Literal[0,1]
  frequence_deplacement: Literal[1,2,3]
  niveau_education: Literal[1,2,3,4,5]
  poste: Literal['Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead']
  statut_marital:Literal['Célibataire','Marié(e)','Divorcé(e)']
  annees_dans_l_entreprise: int
  nombre_experiences_precedentes : int
  annees_dans_le_poste_actuel: int
  annee_experience_totale: int

bundle = load("app/model/model_HR_prediction_TECHNOVA.joblib")
HR_model = bundle['model']
HR_threshold = bundle['threshold']

def FE_ratio_ancienneté(annees_exp_entreprise,annees_exp_tot):
    annees_exp_entreprise/(1+annees_exp_tot)

def FE_duree_moy_exp_precedentes(annees_exp_tot, annees_exp_entreprise, nb_exp):
    result = (annees_exp_tot - annees_exp_entreprise) / (nb_exp+1)
    return result

def FE_ratio_evolution(annees_poste_actuel,annees_exp_entreprise):
    result = annees_poste_actuel/(1+annees_exp_entreprise)
    return result
    
def FE_reste_plus_longtemps(annees_exp_entreprise,duree_moy_exp_precedentes):
    if annees_exp_entreprise > duree_moy_exp_precedentes:
        return 1 
    else:
        return 0
    

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
        "heure_supplementaires": 1,
        "age": 0,
        "annees_dans_l_entreprise": 0
        "nombre_experiences_precedentes" : 0
        "annees_dans_le_poste_actuel":0
        "annee_experience_totale" : 0
        "FE_cadre": 0,
        "niveau_education": 1,
        "poste": "Assistant de Direction",
        "statut_marital": "Célibataire"
    }

    RETURNS : La probabilité identifié par le modèle et la prédiction en fonction du seuil optimisé
    """
    data_dict = data.dict()
    data_dict['FE_ratio_ancienneté'] = FE_ratio_ancienneté(data_dict['annees_dans_l_entreprise'],data_dict['annee_experience_totale'])
    data_dict['FE_duree_moy_exp_precedentes'] = FE_duree_moy_exp_precedentes(data_dict['annee_experience_totale'],data_dict['annees_dans_l_entreprise'],data_dict['nombre_experiences_precedentes'])
    data_dict['FE_ratio_evolution'] = FE_ratio_evolution(data_dict['annees_dans_le_poste_actuel'],data_dict['annees_dans_l_entreprise'])
    data_dict['FE_reste_plus_longtemps'] = FE_reste_plus_longtemps(data_dict['annees_dans_l_entreprise'],data_dict['FE_duree_moy_exp_precedentes'])

    keys_to_delete = {'annee_experience_totale','annees_dans_l_entreprise','annees_dans_le_poste_actuel',"nombre_experiences_precedentes"}
    data_dict_for_model = {k: v for k,v in data_dict.items() if k not in keys_to_delete}

    df = pd.DataFrame([data_dict_for_model])

    

    proba = HR_model.predict_proba(df)[0][1]
    predict = (proba > HR_threshold)
    return {'probabilité': round(float(proba),3),
            'prédiction': bool(predict)}