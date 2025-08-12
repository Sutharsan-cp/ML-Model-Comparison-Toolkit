import json
from Backend.app import app

def test_ping():
    client = app.test_client()
    res = client.get('/api/ping')
    assert res.status_code == 200
    assert res.get_json()["message"] == "Pong from Backend!"

def test_add_numbers():
    client = app.test_client()
    res = client.post('/api/add', json={"a": 5, "b": 7})
    assert res.status_code == 200
    assert res.get_json()["result"] == 12
