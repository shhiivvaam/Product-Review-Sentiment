from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# Load Models
ml_model = joblib.load('./models/ml_model.pkl')
vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')

dl_model = tf.keras.models.load_model('./models/dl_model.h5')

llm_model = BertForSequenceClassification.from_pretrained('./models/llm_model.pt')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@app.route('/predict/ml', methods=['POST'])
def predict_ml():
    data = request.json['text']
    vectorized_data = vectorizer.transform([data])
    prediction = ml_model.predict(vectorized_data)
    return jsonify({'sentiment': int(prediction[0])})

@app.route('/predict/dl', methods=['POST'])
def predict_dl():
    data = request.json['text']
    sequences = tokenizer.texts_to_sequences([data])
    padded_sequences = pad_sequences(sequences, maxlen=200)
    prediction = dl_model.predict(padded_sequences)
    sentiment = int(prediction[0][0] > 0.5)
    return jsonify({'sentiment': sentiment})

@app.route('/predict/llm', methods=['POST'])
def predict_llm():
    data = request.json['text']
    encodings = tokenizer(data, truncation=True, padding=True, max_length=512, return_tensors='pt')
    outputs = llm_model(**encodings)
    sentiment = torch.argmax(outputs.logits, dim=1).item()
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
