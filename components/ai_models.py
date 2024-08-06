import tensorflow as tf

from keras.models import load_model
from keras.utils import to_categorical

import numpy as np
import streamlit as st
import os
import joblib
def load_ai_models():
    directories = {"40": "assets/40_15", "200": "assets/200_15"}
    models = {}

    for model_key, directory in directories.items():
        if not os.path.exists(directory):
            st.error(f"Directory {directory} not found.")
            continue
        
        models[model_key] = {}
        for file in os.listdir(directory):
            if file.endswith(".h5"):
                model_path = os.path.join(directory, file)
                model_name = file[:-3]  #
                models[model_key][model_name] = load_model(model_path)

    return models

def predict_sign_language(data, models, number_choice):
    if number_choice == "40": size = 260 
    else: size = 314
    results = []
    preprocessed_input = load_and_pad_data(data,size) 
    label_encoder = joblib.load(f'assets/{ "40_15" if number_choice == "40" else "200_15" }/label_encoder.pkl')
    prediction= []
    for model_name, model in models.items():
        predictions = model.predict(preprocessed_input)
        predicted_class = np.argmax(predictions, axis=1)
        predicted_label = label_encoder.inverse_transform(predicted_class)
        results.append((model_name, predicted_label, predictions))
        prediction.append(predicted_class)

    average_predictions = np.mean(prediction, axis=0)
    predicted_classs = np.argmax(predictions, axis=1)
    average_decoded_labels = label_encoder.inverse_transform(predicted_classs)

    results.append(("Average", average_decoded_labels, average_predictions))
    
    return results
def load_and_pad_data(data,size):
    new_video = np.array(data)
    padding_needed = size - new_video.shape[0]
    padded_data = np.pad(new_video, pad_width=((0, padding_needed), (0, 0), (0, 0)), mode='constant')
    reshaped_data = padded_data.reshape(1, size, new_video.shape[1] * new_video.shape[2])
    return reshaped_data