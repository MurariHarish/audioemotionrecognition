import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
from scipy.io import wavfile as wav

import IPython.display as ipd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.exception import CustomException
from src.logger import logging

# from src.utils import save_object


@dataclass
class DataTransformationConfig:
    pass

class FeatureGenerator:
    def __init__(self):
        pass
    
    def noise(self, data, nf):
        try:
            return data + nf * np.random.randn(len(data))
        except Exception as e:
                raise CustomException(e,sys)
    
    def stretch(self, data, stretch_f):
        try:
            return librosa.effects.time_stretch(data, rate = stretch_f)
        except Exception as e:
                raise CustomException(e,sys)
    
    def shift(self, data, sampling_rate, shift_r):
        try:
            shift_distance = np.random.randint(shift_r[0], shift_r[1])
            return np.roll(data, shift_distance)
        except Exception as e:
                raise CustomException(e,sys)
    
    def pitch(self, data, sampling_rate, pitch_f):
        try:
            return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_f)
        except Exception as e:
                raise CustomException(e,sys)

    def extract_features(self, data, sample_rate):
        '''
        This function extracts various features from the audio given as input
        '''
        try:
            audio_feature_ar = np.array([])
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
            stft = np.abs(librosa.stft(data))
            chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            mfcc = np.abs(np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0))
            rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
            mel_S = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=data, sr=sample_rate).T, axis=0)
            spec_bandwidth = np.abs(np.mean(librosa.feature.spectral_bandwidth(y=data, sr=sample_rate).T, axis=0))
            audio_feature_ar = np.hstack((audio_feature_ar, 
                                        chroma_stft,
                                        mfcc,
                                        rms,
                                        mel_S,
                                        zero_crossing_rate,
                                        spec_centroid,
                                        spec_bandwidth))

            return audio_feature_ar
        except Exception as e:
                raise CustomException(e,sys)
    
    def get_features(self, path):
        try:
            # Define the noise parameters
            NOISE_FACTOR = 0.05

            # Define the pitch parameters
            PITCH_FACTOR = 2

            data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    
            # without augmentation
            crema_features_ar = self.extract_features(data, sample_rate=sample_rate)
            crema_data_features = np.array(crema_features_ar)
    
            # data with noise
            noise_data = self.noise(data, NOISE_FACTOR)
            res2 = self.extract_features(noise_data, sample_rate=sample_rate)
            crema_data_features = np.vstack((crema_data_features, res2)) # stacking vertically
    
            # data with pitching
            data_stretch_pitch = self.pitch(data, sample_rate, PITCH_FACTOR)
            res3 = self.extract_features(data_stretch_pitch, sample_rate=sample_rate)
            crema_data_features = np.vstack((crema_data_features, res3)) # stacking vertically
    
            return crema_data_features
        
        except Exception as e:
                raise CustomException(e,sys)
            
    def feature_generator(self,raw_data):
        '''
        This function is responsible for collecting all features and storing in dataframe
        '''
        try:
            X_ar = []
            y_ar = []
            for path, emotion in zip(raw_data.path,raw_data.target_emotion):
                feature = self.get_features(path)
                for f in feature:
                    X_ar.append(f)
                    y_ar.append(emotion)
            X_df = pd.DataFrame(X_ar)
            y_df = pd.DataFrame(y_ar)

            return X_df,y_df

        except Exception as e:
                raise CustomException(e,sys)
            

class DataTransformation(FeatureGenerator):

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.scaler=None
            
    def data_transformer(self, X_train_data):
        '''
        This function performs data transformation
        '''
        try:
            self.scaler = StandardScaler().fit(X_train_data)
            # X_train_scaler = scaler.transform(X_train_data)
            # X_test_scaler = scaler.transform(X_test_data)

            # return X_train_scaler,X_test_scaler
            return self.scaler

        except Exception as e:
                raise CustomException(e,sys)
            
    def get_data_transformer(self,X_data):
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted. Call data_transformer with X_train_data first.")
        
        try:
            X_scaler=self.scaler.transform(X_data)
            return X_scaler

        except Exception as e:
                raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function is responsible for data preprocessing
        '''
        try:
            raw_train = pd.read_csv(train_path)
            raw_test = pd.read_csv(test_path)

            logging.info('Reading Train and Test data completed')

            logging.info('Fetching audio features of Train data')
            X_train_df,y_train_df=self.feature_generator(raw_train)

            logging.info('Fetching audio features of Test data')
            X_test_df,y_test_df=self.feature_generator(raw_test)

            logging.info('Data Transformation Started')
            preprocessor_class=DataTransformation()
            preprocessor_class.data_transformer(X_train_df)
            X_train_scaler=preprocessor_class.data_transformer(X_train_df)
            X_test_scaler=preprocessor_class.data_transformer(X_test_df)
            logging.info('Data Transformation Completed')
            
            return X_train_scaler, X_test_scaler, y_train_df, y_test_df

        except Exception as e:
                raise CustomException(e,sys)
