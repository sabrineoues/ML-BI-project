import numpy as np
import pandas as pd
import joblib

# Charger le modèle
knn_pipeline = joblib.load("ml_app/ml_models/knn.pkl")

# Charger dataset pour fréquences
df = pd.read_csv(".csv")

# Exemple de test
salary = 50000
company_name = "Google"
location = "Tunis"

company_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0)
location_freq = df['Location'].value_counts(normalize=True).get(location, 0)

X = np.array([[salary, company_freq, location_freq]])
prediction = knn_pipeline.predict(X)[0]
proba = knn_pipeline.predict_proba(X)[0]

print("Prediction:", prediction)
print("Confidence:", max(proba) * 100)
