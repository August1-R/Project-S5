import librosa
import os
import keras
import numpy as np
from keras.models import load_model

# 1. Définir le nom du fichier
model_filename = 'model_audio.keras'

# 2. Charger le modèle


try:
    loaded_model = load_model(model_filename)
    print(f"Modèle '{model_filename}' chargé avec succès.")
except Exception as e:
    print(f"Erreur lors du chargement du modèle : {e}")
    loaded_model = None



path = "audio_database/Angry/03-01-05-01-01-01-01.wav"
MAX_LEN = 173

X, sample_rate = librosa.load(path, res_type='kaiser_fast')

mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60)

if mfccs.shape[1] > MAX_LEN:
    mfccs = mfccs[:, :MAX_LEN]
else:
    pad_width = MAX_LEN - mfccs.shape[1]
    mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

# CORRECTION : Transposition pour avoir la forme (temps, features) -> (173, 40)
X_test = mfccs.T

X_test = np.expand_dims(X_test, axis=0)



# 3. Utiliser le modèle chargé pour faire des prédictions
if loaded_model:
    # Utilisez loaded_model à la place de 'model' pour la prédiction
    predictions = loaded_model.predict(X_test)

    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index]

    print(f"Prédictions (Probabilités) : {predictions}")
    print(f"Classe prédite (Index) : {predicted_class_index}")
    print(f"Confiance : {confidence:.4f}")