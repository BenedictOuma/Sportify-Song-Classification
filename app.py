#Necessary libraries
import numpy as np
import pandas as pd
import streamlit as st
import joblib

#Loading saved components
model = joblib.load('kmeans_cluster_model.pkl')
label_encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca_.pkl')
labels = joblib.load('cluster_labels.pkl')

#App title and description
st.title('Sportify Songs Playlist Classification')
st.write('This app analyzes your Spotify playlists and classifies songs into various genres or categories. By leveraging machine learning algorithms, it helps you understand the composition of your playlists, discover new music trends, and organize your songs more effectively. Upload your playlist data using the below features or connect your Spotify account to get personalized insights and classifications.')

#Feature input
popularity = st.sidebar.number_input('**Popularity:** (Ranges from 0.0 to 99.0)', 0.0, 99.0)
duration_ms = st.sidebar.number_input('**Duration:** (Ranges from 43235 to 392638)', 43235, 392638)
explicit = st.sidebar.number_input('Explicit Content (0 for False, 1 for True):', min_value = 0, max_value = 1)
danceability = st.sidebar.number_input('**Dance Feel:** (Ranges from 0.06 to 0.99)', 0.06, 0.99)
energy = st.sidebar.number_input('**Energy Level:** (Ranges from 0.01 to 1.00)', 0.01, 1.00)
key = st.sidebar.number_input('**Music Key:** (Ranges from 0.0 to 11.0)', 0.0, 11.0)
loudness = st.sidebar.number_input('**Loudness:** (Ranges from -17.9 to 1.7)', -17.9, 1.7)
mode = st.sidebar.number_input('**Mode:** (Ranges from 0.0 to 1.0)', 0.0, 1.0)
speechiness = st.sidebar.number_input('**Speechiness:** (Ranges from 0.0 to 1.0)', 0.0, 1.0)
acousticness = st.sidebar.number_input('**Acoustic Feel:** (Ranges from 0.0 to 1.0)', 0.0, 1.0)
instrumentalness = st.sidebar.number_input('**Instrumental:** (Ranges from 0.0 to 0.1)', 0.0, 0.1)
liveness = st.sidebar.number_input('**Live Feel:** (Ranges from 0.009 to 0.534)', 0.009, 0.534)
valence = st.sidebar.number_input('**Mood:** (Ranges from 0.000 to 0.994)', 0.000, 0.994)
tempo = st.sidebar.number_input('**Speed:** (Ranges from 36.542 to 209.143)', 36.542, 209.143)
time_signature = st.sidebar.number_input('**Time Signature:** (Ranges from 0.0 to 5.0)', 0.0, 5.0)

#Scaling input features in preparation for the model
features = np.array([[popularity, duration_ms, explicit, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence,	tempo, time_signature]])
scaled = scaler.transform(features)

#Prediction
if st.button('Classify the Song'):
    reduced = pca.transform(scaled)
    cluster = model.predict(reduced)[0]

    cluster_labels = {
        0: "Chill Vibes",
        1: "High Energy Dance",
        2: "Acoustic Mellow"
    }
    prediction_label = cluster_labels.get(cluster, "Unknown Category")

    st.success(f'Classify the Song: {prediction_label}')