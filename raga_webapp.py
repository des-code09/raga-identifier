import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model
import joblib
import tempfile
import os
from pytube import YouTube
from pydub import AudioSegment

# Load model and encoder once
model = load_model("raga_model.h5")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def predict_raga(file_path):
    features = extract_features(file_path)
    prediction = model.predict(np.array([features]))
    return label_encoder.inverse_transform([np.argmax(prediction)])[0]

def process_youtube(link):
    with tempfile.TemporaryDirectory() as tmpdir:
        yt = YouTube(link)
        stream = yt.streams.filter(only_audio=True).first()
        downloaded = stream.download(output_path=tmpdir, filename="audio.mp4")
        audio = AudioSegment.from_file(downloaded)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio = audio[:30000]  # 30 sec
        wav_path = os.path.join(tmpdir, "clip.wav")
        audio.export(wav_path, format="wav")
        return wav_path

# UI
st.title("🎵 Raga Identifier")
st.write("Upload audio, paste a YouTube link, or record to identify the raga.")

option = st.radio("Choose Input Type", ["Upload Audio", "Paste YouTube Link"])

if option == "Upload Audio":
    uploaded_file = st.file_uploader("Upload a .wav file", type=["wav"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            tmpfile.write(uploaded_file.read())
            tmpfile_path = tmpfile.name

        st.audio(uploaded_file, format="audio/wav")
        try:
            raga = predict_raga(tmpfile_path)
            st.success(f"🎼 Predicted Raga: **{raga}**")
        except Exception as e:
            st.error(f"Error: {e}")

elif option == "Paste YouTube Link":
    yt_link = st.text_input("Paste YouTube link:")
    if yt_link:
        try:
            st.info("Downloading and processing...")
            wav_path = process_youtube(yt_link)
            raga = predict_raga(wav_path)
            st.success(f"🎼 Predicted Raga: **{raga}**")
        except Exception as e:
            st.error(f"Error: {e}")
