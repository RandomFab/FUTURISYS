from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_route():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message" : "Bienvenue dans L'API Futurisys. Accédez tout de suite au swagger : https://randomfab-futurisys.hf.space/docs"}