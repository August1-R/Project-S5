import librosa
import os
import time
import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

# --- 1. Paramètres et chargement des données ---

# Chemin vers la base de données audio
path = "audio_database/"
# Longueur fixe pour tous les extraits (en nombre de "frames" MFCC)
MAX_LEN = 173
# Liste pour stocker les données (mfccs, label)
lst = []

model_filename = "model_audio.keras"

print("Début du chargement et du prétraitement des données...")
start_time = time.time()

# Parcours des dossiers pour trouver les fichiers audio
# L'index 'i' servira de label pour chaque classe (dossier)
i = -1
for subdir, dirs, files in os.walk(path):
    # On ignore le dossier racine, on ne traite que les sous-dossiers
    if subdir == path:
        continue

    i += 1
    print(f"Traitement du dossier '{os.path.basename(subdir)}' (Label: {i})")
    for file in files:
        try:
            # Chargement du fichier audio
            X, sample_rate = librosa.load(os.path.join(subdir, file), res_type='kaiser_fast')

            # CORRECTION : Calcul des MFCCs avec la bonne fonction
            mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=60)

            # Padding ou troncature pour avoir une longueur fixe
            if mfccs.shape[1] > MAX_LEN:
                mfccs = mfccs[:, :MAX_LEN]
            else:
                pad_width = MAX_LEN - mfccs.shape[1]
                mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

            # CORRECTION : Transposition pour avoir la forme (temps, features) -> (173, 40)
            arr = mfccs.T, i
            lst.append(arr)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file}: {e}")

print(f"--- Données chargées. Temps de chargement : {time.time() - start_time:.2f} secondes ---")

# --- 2. Préparation des données pour le modèle ---

# Séparation des features (X) et des labels (y)
X, y = zip(*lst)

# Conversion en arrays NumPy
X = np.asarray(X)
y = np.asarray(y)

print(f"Forme du jeu de données X: {X.shape}")  # Devrait être (nb_samples, 173, 40)
print(f"Forme du jeu de données y: {y.shape}")

# Séparation en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 3. Définition du modèle CNN ---

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),

    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.3),  # Un peu de dropout ici

    keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Dropout(0.4),  # Un peu plus de dropout

    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),  # Dropout fort avant la sortie

    keras.layers.Dense(np.unique(y).size, activation='softmax')
])

model.summary()

# --- 4. Compilation et entraînement ---

# OPTIMISATION : Taux d'apprentissage plus élevé
optimiser = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# OPTIMISATION : Callback pour arrêter l'entraînement au bon moment
# Il arrêtera l'entraînement si la 'val_loss' ne s'améliore pas pendant 15 époques
# et restaurera les poids du meilleur modèle trouvé.
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# Entraînement avec moins d'époques (EarlyStopping se chargera d'arrêter)
cnnhistory = model.fit(X_train, y_train,
                       batch_size=32,
                       epochs=100,
                       validation_data=(X_test, y_test),
                       callbacks=[early_stopping])

# --- 5. Évaluation et visualisation ---

# Affichage des courbes d'apprentissage
plt.figure(figsize=(12, 5))

# Graphique de la loss
plt.subplot(1, 2, 1)
plt.plot(cnnhistory.history['loss'], label='Train Loss')
plt.plot(cnnhistory.history['val_loss'], label='Test Loss')
plt.title('Évolution de la Perte (Loss)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

# Graphique de l'accuracy
plt.subplot(1, 2, 2)
plt.plot(cnnhistory.history['accuracy'], label='Train Accuracy')
plt.plot(cnnhistory.history['val_accuracy'], label='Test Accuracy')
plt.title('Évolution de la Précision (Accuracy)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.show()

# Génération du rapport de classification détaillé
print("\n--- Rapport de classification ---")
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# S'assurer que les types sont corrects pour le rapport
y_test = y_test.astype(int)
report = classification_report(y_test, predicted_classes)
print(report)



# --- 5.5. Sauvegarde du modèle ---
model.save(model_filename)
print(f"\nModèle sauvegardé sous le nom : {model_filename}")