import os
from flask import Flask, render_template, request, jsonify
import wave
import librosa
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/findyourmood', methods=['POST'])
def predict_datapoint():
    audiofile = request.files['audiofile']
    if audiofile:
        # Save the audio file temporarily
        filepath = os.path.join("uploads", "temp_audio.wav")
        audiofile.save(filepath)
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict_emotion(filepath)  # predict the mood
        os.remove(filepath)  # remove the temporary audio file
        return jsonify(results[0])

    return jsonify(error="No audio file received"), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

