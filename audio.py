import pyaudio
import librosa
import time
import numpy as np
from keras.models import load_model


class AudioHandler(object):
    def __init__(self):
        self.model_filename = 'model_audio.keras'
        try:
            self.loaded_model = load_model(self.model_filename)
            print(f"Modèle '{self.model_filename}' chargé avec succès.")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle : {e}")
            self.loaded_model = None

        self.MAX_LEN = 173
        self.mffccs = None

        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 88370
        self.p = None
        self.stream = None

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)


        self.mfccs = librosa.feature.mfcc(y=numpy_array, sr=self.RATE, n_mfcc=60)

        X_test = self.mfccs.T

        X_test = np.expand_dims(X_test, axis=0)

        # 3. Utiliser le modèle chargé pour faire des prédictions
        if self.loaded_model:
            # Utilisez loaded_model à la place de 'model' pour la prédiction
            predictions = self.loaded_model.predict(X_test)

            predicted_class_index = np.argmax(predictions, axis=1)[0]
            confidence = predictions[0][predicted_class_index]

            print(f"Prédictions (Probabilités) : {predictions}")
            print(f"Classe prédite (Index) : {predicted_class_index}")
            print(f"Confiance : {confidence:.4f}")

        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()): # if using button you can set self.stream to 0 (self.stream = 0), otherwise you can use a stop condition
            time.sleep(2.0)



audio = AudioHandler()
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()




"""

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


"""