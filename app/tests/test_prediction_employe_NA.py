from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
def test_prediction_employe_NA():
    response = client.post('/predict_from_db_employe',params={'id_employe' : 3})
    assert response.status_code == 200
    result = response.json()
    assert result['message'] =="Aucun employé trouvé"