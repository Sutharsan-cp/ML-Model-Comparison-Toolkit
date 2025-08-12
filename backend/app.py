from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from React frontend

@app.route('/api/ping', methods=['GET'])
def ping():
    return jsonify({"message": "Pong from Backend!"})

@app.route('/api/add', methods=['POST'])
def add_numbers():
    data = request.get_json()
    a = data.get("a", 0)
    b = data.get("b", 0)
    return jsonify({"result": a + b})

if __name__ == '__main__':
    app.run(debug=True)
