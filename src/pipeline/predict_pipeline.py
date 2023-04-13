import sys
import pandas as pd
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler

from src.exception import CustomException
from src.utils import load_object

from src.components.data_transformation import DataTransformation,FeatureGenerator

class PredictPipeline:
    def __init__(self):
        pass
    
    # def extract_features(self, data, sample_rate):
    #     '''
    #     This function extracts various features from the audio given as input
    #     '''
    #     try:
    #         audio_feature_ar = np.array([])
    #         zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    #         stft = np.abs(librosa.stft(data))
    #         chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    #         mfcc = np.abs(np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0))
    #         rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    #         mel_S = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    #         spec_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
    #         spec_bandwidth = np.abs(np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0))
    #         audio_feature_ar = np.hstack((audio_feature_ar, 
    #                                     chroma_stft,
    #                                     mfcc,
    #                                     rms,
    #                                     mel_S,
    #                                     zero_crossing_rate,
    #                                     spec_centroid,
    #                                     spec_bandwidth))

    #         return audio_feature_ar
    #     except Exception as e:
    #             raise CustomException(e,sys)


    def predict_emotion(self,audio_file):
        MODEL_PATH = "artifacts/model.pkl"
        model=load_object(MODEL_PATH)
        # Extract features from audio file using librosa
        data, sample_rate = librosa.load(audio_file)
        features=FeatureGenerator.extract_features(data,sample_rate)
        features=DataTransformation.get_data_transformer(features)
        features=features.to_numpy()
        # features = self.extract_features(data,sample_rate)
        # Predict emotion using the trained model
        prediction = model.predict(features)
        return prediction