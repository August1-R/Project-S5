import keras
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


# 3. Utiliser le modèle chargé pour faire des prédictions
if loaded_model:
    # Utilisez loaded_model à la place de 'model' pour la prédiction
    predictions = loaded_model.predict(X_test)
    # ... le reste de votre logique de prédiction/évaluation ...