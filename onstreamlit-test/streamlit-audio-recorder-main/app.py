import io
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from pydub import AudioSegment
from st_audiorec import st_audiorec
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import tempfile
import os
import matplotlib.pyplot as plt
import h5py

st.set_page_config(page_title="Heart Sound Recorder", page_icon="🎙️")

# Google Drive setup
SERVICE_ACCOUNT_FILE = '/Users/korndanai/Downloads/NSM-INDO-TEST-main-2/onstreamlit-test/streamlit-audio-recorder-main/heart-d9410-9a288317e3c7.json'
SCOPES = ['https://www.googleapis.com/auth/drive']
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def download_file_from_google_drive(file_id, destination):
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(destination, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

GOOGLE_DRIVE_MODEL_FILE_ID = '1A2VnaPoLY3i_LakU1Y_9hB2bWuncK37X'
GOOGLE_DRIVE_LABELS_FILE_ID = '1zIMcBrAi4uiL4zOVU7K2tvbw8Opcf5cW'
MODEL_FILE_PATH = '/Users/korndanai/Downloads/NSM-INDO-TEST-main-2/onstreamlit-test/streamlit-audio-recorder-main/my_model.h5'
LABELS_FILE_PATH = '/Users/korndanai/Downloads/NSM-INDO-TEST-main-2/onstreamlit-test/streamlit-audio-recorder-main/labels.csv'

# Attempt to download files


# Function to load the model with error handling
def load_model():
    try:
        if os.path.exists(MODEL_FILE_PATH):
            model = tf.keras.models.load_model(MODEL_FILE_PATH, custom_objects=None, compile=False)
            return model
        else:
            st.error(f"Model file not found at {MODEL_FILE_PATH}.")
            return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the pre-trained model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please ensure the model file exists and is correct.")
    if st.button('Retry Loading Model'):
        model = load_model()
        st.experimental_rerun()

# Initialize the encoder
encoder = LabelEncoder()
labels = pd.read_csv(LABELS_FILE_PATH)
encoder.fit(labels['label'])

def extract_heart_sound(audio):
    fourier_transform = np.fft.fft(audio)
    heart_sound = np.abs(fourier_transform)
    return heart_sound

def preprocess_audio(file, file_format):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp_file:
            temp_file.write(file.read())
            temp_file.flush()
            temp_file_path = temp_file.name
        
        # Convert audio to WAV format if necessary
        audio = None
        if file_format in ['mp3', 'm4a', 'x-m4a', 'ogg', 'flac', 'aac', 'wma']:
            audio = AudioSegment.from_file(temp_file_path, format=file_format)
            temp_wav_path = temp_file_path.replace(f".{file_format}", ".wav")
            audio.export(temp_wav_path, format='wav')
        else:
            temp_wav_path = temp_file_path
        
        # Load the audio file using librosa
        y, sr = librosa.load(temp_wav_path, sr=None)
        # Normalize the audio
        audio = y / np.max(np.abs(y))
        # Extract heart sound using Fourier transform
        heart_sound = extract_heart_sound(audio)
        # Generate the spectrogram
        spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram)
        # Define a fixed length for the spectrogram
        fixed_length = 1000  # Adjust this value as necessary
        # Pad or truncate the spectrogram to the fixed length
        if spectrogram.shape[1] > fixed_length:
            spectrogram = spectrogram[:, :fixed_length]
        else:
            padding = fixed_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, padding)), 'constant')
        # Reshape the spectrogram to fit the model
        spectrogram = spectrogram.reshape((1, 128, 1000, 1))
        return spectrogram
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None
    finally:
        # Clean up the temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

# Streamlit interface for recording and uploading audio files
st.markdown('''
    <style>
        .css-1egvi7u {margin-top: -3rem;}
        .stAudio {height: 45px;}
        .css-v37k9u a, .css-nlntq9 a {color: #ff4c4b;}
        .header {background-color: #b71c1c; color: white; padding: 10px;}
        .title {font-size: 30px; margin-bottom: 10px;}
        .waveform {background-color: #f0f0f0; padding: 20px; border-radius: 5px;}
        .progress-bar {margin-top: 20px;}
    </style>
''', unsafe_allow_html=True)
st.markdown('<div class="header"><div class="title">Heart Sound Recorder</div></div>', unsafe_allow_html=True)

recording_status = st.empty()
if 'recording' not in st.session_state:
    st.session_state['recording'] = False
if st.session_state['recording']:
    recording_status.markdown('<div class="waveform">Recording...</div>', unsafe_allow_html=True)
else:
    recording_status.markdown('<div class="waveform">Click to record</div>', unsafe_allow_html=True)

wav_audio_data = st_audiorec()
uploaded_file = st.file_uploader("Choose a file (WAV, MP3, M4A, FLAC, AAC, OGG, WMA)", type=['wav', 'mp3', 'm4a', 'x-m4a', 'flac', 'aac', 'ogg', 'wma'])

audio_data = None
file_format = None

if wav_audio_data is not None:
    audio_data = io.BytesIO(wav_audio_data)
    file_format = 'wav'
    st.audio(wav_audio_data, format='audio/wav')
elif uploaded_file is not None:
    audio_data = uploaded_file
    file_format = uploaded_file.type.split('/')[1]
    st.audio(uploaded_file, format=f'audio/{file_format}')

if audio_data is not None:
    progress_text = st.empty()
    progress_text.text("Recording complete. Click the button below to get the prediction.")
    if st.button('Diagnose'):
        with st.spinner('Uploading audio and getting prediction...'):
            spectrogram = preprocess_audio(audio_data, file_format)
            if spectrogram is not None:
                # Get prediction probabilities
                y_pred = model.predict(spectrogram)
                class_probabilities = y_pred[0]
                sorted_indices = np.argsort(-class_probabilities)  # Sorted indices of classes in descending order
                predicted_label = encoder.inverse_transform([sorted_indices[0]])[0]
                confidence_score = class_probabilities[sorted_indices[0]]
                # Handle case where artifact is 100% confidence
                if predicted_label == 'artifact' and confidence_score >= 0.70:
                    st.write("Artifact detected with high confidence. Please try recording again due to too many noises.")
                    st.write(f"Prediction: artifact")
                    st.write(f"Confidence: {confidence_score:.2f}")
                else:
                    if predicted_label == 'artifact':
                        predicted_label = encoder.inverse_transform([sorted_indices[1]])[0]
                        confidence_score = class_probabilities[sorted_indices[1]]
                    st.write(f"Prediction: {predicted_label}")
                    st.write(f"Confidence: {confidence_score:.2f}")
                    # Plot the class probabilities
                    fig, ax = plt.subplots()
                    ax.bar(encoder.classes_, class_probabilities, color='blue')
                    ax.set_xlabel('Class')
                    ax.set_ylabel('Probability')
                    ax.set_title('Class Probabilities')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
                    # Show accuracy of all classes in a collapsible section
                    with st.expander("Show Class Accuracies"):
                        for i, label in enumerate(encoder.classes_):
                            st.write(f"Accuracy for class '{label}': {class_probabilities[i]:.2f}")
            else:
                st.error("Failed to process the audio.")