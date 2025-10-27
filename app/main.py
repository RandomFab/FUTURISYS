# app/main.py
from fastapi import FastAPI,Query
from joblib import load
from pydantic import BaseModel
from typing import Literal
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
from .utils.connexion_db import connexion_db
from .utils.feature_engineering import transform_fe
from .utils.interaction_db import get_employe, post_input, post_output


app = FastAPI()

class PredictionRawData(BaseModel):
  heure_supplementaires: Literal[0,1]
  age: int
  frequence_deplacement: Literal[0,1,2]
  niveau_education: Literal[1,2,3,4,5]
  poste: Literal['Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead']
  statut_marital:Literal['Célibataire','Marié(e)','Divorcé(e)']
  annees_dans_l_entreprise: int
  nombre_experiences_precedentes : int
  annees_dans_le_poste_actuel: int
  annee_experience_totale: int

class PredictionTransformedData(BaseModel):
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


    
@app.get("/")
def read_root():
    return {"message": "Bienvenue dans L'API Futurisys. Accédez tout de suite au swagger : https://randomfab-futurisys.hf.space/docs"}

@app.get('/threshold')
def get_threshold():
    """
    Retourne le seuil du f1 score optimal du modèle 
    """
    return HR_threshold

@app.get("/features")
def get_features():
    """
    Retourne les variables utilisés par le modèle pour calculé la prédiction
    """
    preprocessor = HR_model.named_steps['preprocessing']

    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]

    all_features = list(num_features) + list(cat_features)

    return all_features

@app.get('/model-info')
def get_model_info():
    """
    Retourne toutes les informations du modèle (type, valeurs des hyperparamètres, variables utilisées par le modèle, seuil f1 optimisé)
    """
    model = HR_model.named_steps['model']
    infos_model = {
        'type' : model.__class__.__name__,
        'params' : model.get_params(),
        'feature_names' : get_features(),
        'threshold_F1_optimised' : get_threshold()
    }
    return infos_model

@app.post('/predict_from_raw_data')
def post_prediction_from_raw_data(data: PredictionRawData):

    """
    Prédit le départ ou non d'un employé sur la base de données brutes accessibles par les RH

    ARGS : un dictionnaire contenant les valeurs des différentes variables 
    {
        "heure_supplementaires": 0,1\n
        "age": 0-100\n
        "annees_dans_l_entreprise": int\n
        "frequence_deplacement": 0(peu), 1(occasionnel), 2(fréquent)\n
        "nombre_experiences_precedentes" : int\n
        "annees_dans_le_poste_actuel":int\n
        "annee_experience_totale" : int\n
        "niveau_education": 1,2,3,4,5\n
        "poste": 'Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead',\n
        "statut_marital": 'Célibataire','Marié(e)','Divorcé(e)'\n
    }

    RETURNS : La probabilité identifié par le modèle et la prédiction en fonction du seuil optimisé
    """
    data_dict_for_model = transform_fe(data.dict())

    df = pd.DataFrame([data_dict_for_model])

    proba = HR_model.predict_proba(df)[0][1]
    predict = (proba > HR_threshold)
    return {'probabilité': round(float(proba),3),
            'prédiction': bool(predict)}

@app.post('/predict_from_transformed_data')
def post_prediction_from_transformed_data(data: PredictionTransformedData):

    """
    Prédit le départ ou non d'un employé sur la base de données déjà calculer (Feature engineering)

    ARGS : un dictionnaire contenant les valeurs des différentes variables 
    {
        "heure_supplementaires": 0,1\n
        "age": 0-100\n
        "FE_ratio_ancienneté": float\n
        "FE_cadre": 0,1\n
        "frequence_deplacement": 0(peu), 1(occasionnel), 2(fréquent)\n
        "FE_duree_moy_exp_precedentes": float\n
        "FE_ratio_evolution": float\n
        "niveau_education": 1,2,3,4,5\n
        "FE_reste_plus_longtemps": 0,1\n
        "poste": 'Assistant de Direction','Cadre Commercial','Consultant','Directeur Technique','Manager','Représentant Commercial','Ressources Humaines','Senior Manager','Tech Lead',\n
        "statut_marital": 'Célibataire','Marié(e)','Divorcé(e)'\n
    }

    RETURNS : La probabilité identifié par le modèle et la prédiction en fonction du seuil optimisé
    """

    df = pd.DataFrame([data.dict()])
    proba = HR_model.predict_proba(df)[0][1]
    predict = (proba > HR_threshold)
    return {'probabilité': round(float(proba),3),
            'prédiction': bool(predict)}


@app.post('/predict_from_db_employe')
def post_prediction_from_raw_data(id_employe: int = Query(..., description="identifiant de l'employé", ge=1)):

    """
    Prédit le départ ou non d'un employé sur la base des données de la DB employe

    ARGS : un dictionnaire contenant les valeurs des différentes variables 
    {
        id_employee
    }

    RETURNS : La probabilité identifié par le modèle et la prédiction en fonction du seuil optimisé
    """
    engine = connexion_db()

    with engine.connect() as conn:
        #FBL: Récupération des données employées
        data_dict = get_employe(conn,id_employe)
        if "message" in data_dict:
            return data_dict

    data_dict_for_model = transform_fe(data_dict)

    df = pd.DataFrame([data_dict_for_model])

    proba = round(HR_model.predict_proba(df)[0][1],3)
    predict = (proba > HR_threshold)

    with engine.begin() as conn:
        #FBL: Ajouter l'input à inputs db
        id_input = post_input(conn,data_dict_for_model)
        #FBL: Ajouter les résultats à output
        post_output(conn, id_input, proba, predict)

    return {'params' : data_dict,
            "message": "ligne insérée avec succès dans inputs",
        'probabilité': round(float(proba),3),
        'prédiction': bool(predict)}