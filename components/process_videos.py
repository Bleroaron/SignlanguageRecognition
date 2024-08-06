import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import os
import mediapipe as mp
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import plotly.express as px
from mediapipe.python.solutions.holistic import Holistic
import logging
from io import BytesIO
import plotly.graph_objects as go

dtype = [('left_data', np.float64, (21, 3)),
          ('left_truthy', np.bool_),
          ('right_data', np.float64, (21, 3)),
          ('right_truthy', np.bool_),
          ('mouth_data', np.float64, (6, 3)),
          ('mouth_truthy', np.bool_),
          ('frame_truthy', np.bool_)]

dtype2 = [('left_data', np.float64, (21, 3)),
          ('left_truthy', np.bool_),
          ('right_data', np.float64, (21, 3)),
          ('right_truthy', np.bool_),
          ('mouth_data', np.float64, (6, 3)),
          ('mouth_truthy', np.bool_),
]

dtype3 = [('left_data', np.float64, (21, 3)),
          ('right_data', np.float64, (21, 3)),
          ('mouth_data', np.float64, (6, 3)),
]

def display_images_grid(images, cols=10):
    num_images = len(images)
    num_rows = (num_images + cols - 1) // cols
    fig, axs = plt.subplots(num_rows, cols, figsize=(20, num_rows + 1))
    axs = axs.flatten()
    
    for i, ax in enumerate(axs):
        if i < num_images:
            ax.imshow(images[i])  
            ax.axis('off')
        else:
            ax.axis('off')  
    
    plt.tight_layout(pad=0.1)
    
    # Convert the matplotlib plot to an image that can be displayed by Streamlit
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    
    return buf


def _3D_visualisation(scaled_mapped_video, start=None, end=None):
    data = np.array([np.concatenate([frame[0], frame[1], frame[2]]) for frame in scaled_mapped_video])
    num_frames = data.shape[0]
    x_min, y_min = data[:, :, 0].min(), data[:, :, 1].min()  # Minimum for x and y
    x_max, y_max = data[:, :, 0].max(), data[:, :, 1].max()  # Maximum for x and y
    
    fig = go.Figure()

    # Adding all frames to the figure but make them invisible initially
    for i in range(num_frames):
        fig.add_trace(
            go.Scatter3d(
                x=data[i, :, 0],
                y=data[i, :, 1],
                z=data[i, :, 2],
                mode='markers',
                marker=dict(size=2),
                visible=(i == 0)  # only the first frame is visible
            )
        )

    # Define frames for animation
    frames = [dict(name=str(k),
                   data=[dict(type='scatter3d',
                              x=data[k, :, 0],
                              y=data[k, :, 1],
                              z=data[k, :, 2])],
                   traces=[0])
              for k in range(num_frames)]

    fig.frames = frames
    

    # Add play button
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                },
            ]
        }],
        scene=dict(
            xaxis=dict(nticks=10, range=[x_min, x_max]),
            yaxis=dict(nticks=10, range=[y_min, y_max]),
            zaxis=dict(nticks=10, range=[-1,1]),
            aspectmode='cube',  # This maintains equal aspect ratio across all axes
            camera = dict(
                up=dict(x=0, y=0, z=0),  # This sets the upward direction along the y-axis
                center=dict(x=0, y=0, z=0),  # Centers the camera on the plot
                eye=dict(x=0, y=0, z=1.5)  # Position the camera above the plot
        )
        )
    )

    return fig


def clean_image_data(frame_data):
    #Remove falsy images
    cleaned_data = []
    for i in range(len(frame_data)):
        frame, is_truthy = frame_data[i]
        if is_truthy:
            cleaned_data.append(frame)
        else:
            # Check if any subsequent frame is truthy
            future_truthy = any(truth for _, truth in frame_data[i+1:])
            if future_truthy:
                cleaned_data.append(frame)
    return cleaned_data

def clean_frame_data(frame_data):
    #Remove falsy frames
    cleaned_data = []
    for i in range(len(frame_data)):
        frame = frame_data[i]
        if frame['frame_truthy']:
            cleaned_data.append(frame)
        else:
            future_truthy = any(frame_data[j]['frame_truthy'] for j in range(i + 1, len(frame_data)))
            if future_truthy:
                cleaned_data.append(frame)
    new_cleaned_data = np.array([(
        frame['left_data'], frame['left_truthy'],
        frame['right_data'], frame['right_truthy'],
        frame['mouth_data'], frame['mouth_truthy']
    ) for frame in cleaned_data], dtype=dtype2)

    return new_cleaned_data


def to_Horizontal(frame_data):
    left = [(frame['left_data'], frame['left_truthy']) for frame in frame_data]
    right = [(frame['right_data'], frame['right_truthy']) for frame in frame_data]
    mouth = [(frame['mouth_data'], frame['mouth_truthy']) for frame in frame_data]
    return [left, right, mouth]




def to_Vertical(horizontal_data):
    num_frames = len(horizontal_data[0])
    structured_data = np.zeros(num_frames, dtype=dtype3)
    for i in range(num_frames):
        structured_data[i]['left_data'] = horizontal_data[0][i]
        structured_data[i]['right_data'] = horizontal_data[1][i]
        structured_data[i]['mouth_data'] = horizontal_data[2][i]
    return structured_data



def interpolate(start_data, end_data, segments):
    points = []
    for i in range(1, segments):
        t = i / (segments + 1)
        interpolated_point = (1 - t) * np.array(start_data) + t * np.array(end_data)
        points.append(interpolated_point)
    return np.array(points)


def find_surrounded_false_sequences(data):
    surrounding_true_indices = []
    false_sequence = False
    start_index = None
    for i, (_, truthy) in enumerate(data):
        if not truthy:
            # If current value is False start tracking
            if not false_sequence:
                false_sequence = True
                start_index = i
        else:
            # Check if its a false sequence
            if false_sequence:
                # Ensure the False sequence is surrounded by True
                if start_index > 0 and i < len(data) and data[start_index - 1][1] and data[i][1]:
                    surrounding_true_indices.append((start_index - 1, i))
                false_sequence = False
    return surrounding_true_indices


def process_landmarks(data):
    cleaned_data = clean_frame_data(data)
    horizontal_data = to_Horizontal(cleaned_data)
    result = []
    for i,data in enumerate(horizontal_data):

        body_part_results = []
        all_true = all(truthy == True for _,truthy in data)
        all_false = all(truthy == False for _,truthy in data)
        grouped_true = bool(re.match('^0*1+0*$',''.join('1' if truthy else '0' for _,truthy in data)))

        for array, _ in data:
            body_part_results.append(array)
        if all_true or grouped_true or all_false:
            a=0

        else:
            indices = find_surrounded_false_sequences(data)
            for start_idx, end_idx in indices:
              # Check if there is space between indices to interpolate
              if end_idx - start_idx > 1:
                  # Generate interpolated values
                  interpolated_array = interpolate(data[start_idx][0], data[end_idx][0], end_idx - start_idx)
                  # Replace the values in the data array
                  for i in range(len(interpolated_array)):
                      body_part_results[start_idx + i + 1] = np.array(interpolated_array[i])
        result.append(body_part_results)

    combined_array = to_Vertical(result)
    return combined_array

def annotate_image(landmarks_images):

  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_holistic = mp.solutions.holistic
  images = []
  for image,landmark in landmarks_images:
      if landmark.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            landmark.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
      if landmark.left_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              landmark.left_hand_landmarks,
              mp_holistic.HAND_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
      if landmark.right_hand_landmarks:
          mp_drawing.draw_landmarks(
              image,
              landmark.right_hand_landmarks,
              mp_holistic.HAND_CONNECTIONS,
              landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style())
      images.append(image)
  return images



def process_hand_landmarks(landmarks):
    """Converts hand landmarks to a NumPy array of shape (21, 3) with float64 precision."""
    if landmarks is None:
        return np.zeros((21, 3), dtype=np.float64), False
    return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float64), True

def process_mouth_landmarks(landmarks):
    indices = [13, 14, 33, 78, 263, 308]
    """Converts facial landmarks to a NumPy array with float64 precision."""
    if landmarks is None:
        return np.zeros((6, 3), dtype=np.float64), False #468
    # return np.array([(lm.x, lm.y, lm.z) for lm in landmarks.landmark], dtype=np.float64), True
    return np.array([(lm.x, lm.y, lm.z) for i, lm in enumerate(landmarks.landmark) if i in indices]), True

def scale_points(points, scale_factor):
    """Scales points about their centroid using float64 precision."""
    points = np.array(points, dtype=np.float64)
    if points.size == 0:
        return points  # Return empty array if no points
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    scaled_points = translated_points * scale_factor
    final_points = scaled_points + centroid
    return final_points

def calculate_distance(point_a, point_b):
    """Calculates the Euclidean distance between two points using float64 precision."""
    return np.linalg.norm(np.array(point_a, dtype=np.float64) - np.array(point_b, dtype=np.float64))

def frame_square(frame):
    height, width = frame.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    left = (width - size) // 2
    return frame[top:top + size, left:left + size]

def process_video(video_file_path):
    frame_data = []
    annotated_frames = []
    start = False
    seen_landmarks = {"left":False,"right":False,"mouth":False}
    frame_number = 0
    cap = cv2.VideoCapture(video_file_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        truthy = True
        square_frame = frame_square(frame)
        frame_rgb = cv2.cvtColor(square_frame, cv2.COLOR_BGR2RGB)
        mp_holistic = mp.solutions.holistic
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            landmarks = holistic.process(frame_rgb)
            left_data, value_l = process_hand_landmarks(landmarks.left_hand_landmarks)
            right_data, value_r = process_hand_landmarks(landmarks.right_hand_landmarks)
            mouth_data, value_m = process_mouth_landmarks(landmarks.face_landmarks)

            if not start and (value_l or value_r):
                start = True

            if start:
                seen_landmarks["left"] = seen_landmarks["left"] or value_l
                seen_landmarks["right"] = seen_landmarks["right"] or value_r
                seen_landmarks["mouth"] = seen_landmarks["mouth"] or value_m


            if start and (not value_l and not value_r):
                    truthy = False

            if start:
                annotated_image = (frame_rgb.copy(),landmarks)
                frame_data.append((left_data,value_l, right_data,value_r, mouth_data,value_m,truthy))
                annotated_frames.append((annotated_image,truthy))
                frame_number += 1


        
    all_data = np.array(frame_data, dtype=dtype)
    annotate_images = np.array(annotated_frames, dtype=object)
    cap.release()
    cv2.destroyAllWindows()

    return all_data, annotate_images


def process_video_wrapper(video_path,reference_distances):
    reference_m_indices=[2,4] 
    reference_h_indices=[0,12]
    frame_data, images_landmarks_array = process_video(video_path)
    processed_data = process_landmarks(frame_data)
    if len(processed_data) !=0:
      pos = len(processed_data) // 2
      def cal_distance(start_pos):
          for pos in range(start_pos, len(processed_data)):
              try:
                  current_distance_l = calculate_distance(processed_data[pos][0][reference_h_indices[0]], processed_data[pos][0][reference_h_indices[1]])
                  current_distance_r = calculate_distance(processed_data[pos][1][reference_h_indices[0]], processed_data[pos][1][reference_h_indices[1]])
                  current_distance_m = calculate_distance(processed_data[pos][2][reference_m_indices[0]], processed_data[pos][2][reference_m_indices[1]])
                  return current_distance_l, current_distance_r, current_distance_m
              except IndexError:
                  continue 
          return [None, None, None]

      
      current_distances = cal_distance(pos)

      # Calculate scale factors if current distances are not zero
      scale_factors = []
      for ref_dist, curr_dist in zip(reference_distances, current_distances):
          if curr_dist == 0 or curr_dist is None:  
              scale_factors.append(1)  
          else:
              scale_factors.append(ref_dist / curr_dist) 

      # Apply scaling and map to video frames
      scaled_mapped_video = [[scale_points(frame[j], scale_factors) for j in range(3)] for frame in processed_data]

      # Clean the images
      clean_images_landmarks_array = clean_image_data(images_landmarks_array)


      return scaled_mapped_video,clean_images_landmarks_array


def standardise(data):
    part_stats = []
    num_parts = len(data[0])

    for part_index in range(num_parts):
        all_z_values = np.concatenate([np.array(entry[part_index])[:, 2] for entry in data])
        global_z_mean = np.mean(all_z_values)
        global_z_std = np.std(all_z_values)
        part_stats.append((global_z_mean, global_z_std))

    standardised_data = []
    for entry in data:
        standardised = []
        for part_index, part in enumerate(entry):
            part = np.array(part)
            global_z_mean, global_z_std = part_stats[part_index]

            standardised_z = np.zeros(part[:, 2].shape)
            if global_z_std != 0:
                standardised_z = (part[:, 2] - global_z_mean) / global_z_std
            else:
                standardised_z = np.zeros(part[:, 2].shape)  # Ensures all standardised values are zero when STD is zero

            # if input z is zero, set standardised z to zero
            standardised_z[part[:, 2] == 0] = 0

            new_part = np.copy(part)
            new_part[:, 2] = standardised_z
            standardised.append(new_part)
        standardised_data.append(standardised)

    return standardised_data




def wrapper(path,visualise,reference_distances):
    data = process_video_wrapper(path,reference_distances)
    if data is not None:
        processed_data, clean_images_landmarks_array = data
        standardised_data = standardise(processed_data)
        annotated_images = None
        if visualise:
            annotated_images = annotate_image(clean_images_landmarks_array)
        new_video = []
        for frame in standardised_data:
            concatenated_frame = np.concatenate(frame, axis=0)
            new_video.append(concatenated_frame)

        return new_video,annotated_images,processed_data
    else:
        return None


