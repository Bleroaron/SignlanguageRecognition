import streamlit as st
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from components.process_videos import wrapper, calculate_distance, display_images_grid, _3D_visualisation
from components.ai_models import predict_sign_language,load_ai_models
from components.words import _40_list,_200_list
import pandas as pd
import joblib
import tempfile

def process_uploads(uploaded_files, reference_distances):
    results = {}
    temp_files = [] 

    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_files.append((temp_file.name, uploaded_file.name))  
        with open(temp_file.name, 'wb') as f:
            f.write(uploaded_file.getvalue())  
        temp_file.close()

    with ProcessPoolExecutor() as executor:
        future_to_video = {executor.submit(wrapper, temp_path, True, reference_distances): original_name for temp_path, original_name in temp_files}
        
        for future in as_completed(future_to_video):
            video_name = future_to_video[future]
            try:
                result = future.result()
                results[video_name] = result if result else ("No result", None, None)
            except Exception as e:
                results[video_name] = f"Error processing video {video_name}: {e}"

    for temp_path, _ in temp_files:
        os.unlink(temp_path)

    return results

def get_reference_distance():
    reference_distances = [0.08876117044589021, 0.1032161793973281, 0.09168395195822147]
    return reference_distances




def get_models(number,model_list):
    model_subset = models.get(number, {})
    selected_models = {name: model_subset[name] for name in model_list if name in model_subset}
    return selected_models


def visualise_data(images,visual):
   
    image_buf = display_images_grid(images)
    fig = _3D_visualisation(visual)
    st.image(image_buf, caption='Image Grid', use_column_width=True)
    st.plotly_chart(fig, use_container_width=True)


models = load_ai_models()  

def main():
    st.title('Sign Language Recognition')

    if 'error_message' not in st.session_state:
        st.session_state.error_message = ""
    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'predicted_results' not in st.session_state:
        st.session_state.predicted_results = {}
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'visualise' not in st.session_state:
        st.session_state.visualise = False

   
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

   
    number_choice = st.radio("Select dictionary size", list(models.keys()))

    col1, col2 = st.columns(2)
    with col1:
        dictionary = st.checkbox("Show dictionary")
    with col2:
        visualise = st.checkbox("Show visualisation")

    if visualise:
        st.session_state.visualise = True
    else:
        st.session_state.visualise = False

    if dictionary:
        if number_choice == "40":
            st.dataframe(pd.DataFrame([_40_list]))
        else:
            st.dataframe(pd.DataFrame([_200_list]))

    selected_models = st.multiselect('Select Models', options=models[number_choice].keys())
    uploaded_files = st.file_uploader("Choose video files", accept_multiple_files=True, type=['mp4'])

    # Process uploads and detect features
    if uploaded_files and selected_models and st.button('Feature Detection'):
        file_list = [uploaded_file.name for uploaded_file in uploaded_files]
        if st.session_state.uploaded_files != file_list:
            st.session_state.uploaded_files = file_list
            st.session_state.results = process_uploads(uploaded_files, get_reference_distance())
            st.session_state.predicted_results = {}
            
    # Visualize data 
    if st.session_state.results and st.session_state.visualise is True:
        for key, result in st.session_state.results.items():
            with st.expander(f"Visualisation for {key}"):
                if isinstance(result, tuple):
                    data, images, visual = result
                    visualise_data(images, visual)
                else:
                    st.write(result)

    # Perform predictions
    if st.session_state.results and st.button('Predict Sign Language'):
        st.session_state.predicted_results = {key: predict_sign_language(data, get_models(number_choice, selected_models),number_choice)
                             for key, (data, images, visual) in st.session_state.results.items()}
        for key, predictions in st.session_state.predicted_results.items():
            with st.expander(f"Prediction(s) for {key}"):
                df = pd.DataFrame([prediction[1] for prediction in predictions], index=[prediction[0] for prediction in predictions])
                st.dataframe(df)


if __name__ == "__main__":
    main()
