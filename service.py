#!flask/bin/python
import os
from flask import Flask
from flask import request
from waitress import serve

from train_model import load_from_pickle
from train_model import preprocess_question
import config

vectorizer = None
classifier = None

app = Flask(__name__)

@app.route(f'{config.URI_BASE}/health-check')
def index():
    return "true"

@app.route(f'{config.URI_BASE}/predictions/specialization', methods=['GET'])
def get_prediction():
    question = request.args.get('question')

    question = preprocess_question(question)
    embedding = vectorizer.transform([question])
    prediction = classifier.predict(embedding)[0]

    return str(prediction)


if __name__ == '__main__':
    vectorizer = load_from_pickle(config.PICKLE_VECTORIZER_NAME)
    classifier = load_from_pickle(config.PICKLE_CLASSIFIER_NAME)

    serve(app, host=config.HOST, port=config.PORT)

