# app/main.py
from fastapi import FastAPI
from joblib import load

app = FastAPI()


bundle = load("app/model/model_HR_prediction_TECHNOVA.joblib")
HR_model = bundle['model']
HR_threshold = bundle['threshold']


@app.get('/threshold')
def get_threshold():
    return HR_threshold

@app.get("/features")
def get_features():
    preprocessor = HR_model.named_steps['preprocessing']
    encoder = preprocessor.named_transformers_['encoder']

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



