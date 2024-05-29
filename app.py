from flask import Flask, request, jsonify, render_template, send_from_directory
from google.cloud import aiplatform
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def get_embedding(input_text):
    endpoint = aiplatform.Endpoint(
        endpoint_name="projects/YOUR_PROJECT/locations/us-central1/endpoints/YOUR_ENDPOINT"
    )
    response = endpoint.predict(
        instances=[{"text": [input_text]}]
    )
    embedding = response.predictions[0]['text'][0]
    return np.array(embedding)

def load_audio():
    with open('audio.json', 'r') as f:
        return json.load(f)

def load_embeddings():
    with open('embeddings.json', 'r') as f:
        return json.load(f)

def find_nearest_neighbor(user_embedding, embeddings):
    user_embedding = user_embedding.reshape(1, -1)
    similarities = {}
    for i, embedding in enumerate(embeddings):
        embedding = np.array(embedding).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, embedding)[0][0]
        similarities[str(i)] = similarity
    nearest_key = max(similarities, key=similarities.get)
    return nearest_key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    input_text = request.form['input_text']
    user_embedding = get_embedding(input_text)
    embeddings = load_embeddings()['predictions'][0]['audio']
    nearest_key = find_nearest_neighbor(user_embedding, embeddings)
    audio = load_audio()
    audio_file = audio['instances'][0]['audio'][int(nearest_key)]
    return jsonify({'audio_file': audio_file})

@app.route('/static/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

if __name__ == '__main__':
    app.run(debug=True)
