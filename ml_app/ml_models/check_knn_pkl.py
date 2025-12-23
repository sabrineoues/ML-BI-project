import joblib
import numpy as np

# Charger le fichier
pipeline = joblib.load("ml_app/ml_models/knn.pkl")

# Vérifier que c'est bien un tuple
print("Type pipeline:", type(pipeline))
print("Longueur pipeline:", len(pipeline))

# Déballer
if isinstance(pipeline, tuple):
    knn_model, scaler, encoder_company, encoder_location = pipeline

    print("knn_model:", type(knn_model))
    print("scaler:", type(scaler))
    print("encoder_company:", type(encoder_company))
    print("encoder_location:", type(encoder_location))

    print("Classes company:", encoder_company.classes_)
    print("Classes location:", encoder_location.classes_)

    # Test rapide du scaler
    sample = np.array([[1, 2, 3000]])
    print("Scaled sample:", scaler.transform(sample))
