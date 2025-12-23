from django.shortcuts import render
from .forms import KNNSampleForm
from django.conf import settings
import joblib
from .forms import XGBForm
import os
import logging
import numpy as np
import pandas as pd
from .ml_utils import _load_model




# Charger le dataset pour calculer les fréquences automatiquement
df = pd.read_csv('C:\\3eme\\datawarehouse\\dataset_ceaned.csv')  # ou depuis la base Django

def knn_view(request):
    form = KNNSampleForm(request.POST or None)
    result = None
    label = None
    confidence = None
    model_missing = False

    try:
        knn_pipeline = _load_model('knn.pkl')
    except FileNotFoundError:
        knn_pipeline = None
        model_missing = True

    if form.is_valid() and knn_pipeline is not None:
        try:
            company_name = form.cleaned_data['company_name']
            location = form.cleaned_data['location']
            salary = form.cleaned_data['salary']

            # Calcul des fréquences automatiquement
            company_name_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0.0001)
            location_freq = df['Location'].value_counts(normalize=True).get(location, 0.0001)

            # Préparer les données dans le bon ordre attendu par le pipeline
            X = np.array([[salary, company_name_freq, location_freq]])

            # Prédiction
            result = knn_pipeline.predict(X)[0]

            # Mapping label
            joblevel_label = {0: 'Junior', 1: 'Medior', 2: 'Senior'}
            label = joblevel_label.get(int(result), str(result))

            # Confiance (%)
            if hasattr(knn_pipeline, 'predict_proba'):
                probs = knn_pipeline.predict_proba(X)[0]
                confidence = round(float(np.max(probs) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction KNN")
            result, label, confidence = None, None, None

    return render(request, 'knn.html', {
        'form': form,
        'result': result,
        'label': label,
        'confidence': confidence,
        'model_missing': model_missing,
    })

def xgb_view(request):
    form = XGBForm(request.POST or None)
    result = None
    label = None
    confidence = None
    model_missing = False

    # Charger le modèle XGBoost
    try:
        xgb_model = _load_model('xgb_model.pkl')
    except FileNotFoundError:
        xgb_model = None
        model_missing = True

    if form.is_valid() and xgb_model is not None:
        try:
            # Récupérer les entrées utilisateur
            location = form.cleaned_data['location']
            skill = form.cleaned_data['skill']
            company_name = form.cleaned_data['company_name']
            platform_name = form.cleaned_data['platform_name']
            degree = form.cleaned_data['degree']

            # Transformation des mots en fréquences / scores
            location_freq = df['Location'].value_counts(normalize=True).get(location, 0)
            skill_score = df['skill_score'].value_counts(normalize=True).get(skill, 0)
            company_freq = df['company_name'].value_counts(normalize=True).get(company_name, 0)
            platform_freq = df['platform_name'].value_counts(normalize=True).get(platform_name, 0)
            # Pour degree, encode avec factorize
            degree_encoded = pd.factorize(df['degree'])[1].get_loc(degree) if degree in df['degree'].values else 0

            # Préparer les données dans l'ordre exact attendu par le modèle
            X = np.array([[location_freq, skill_score, company_freq, platform_freq, degree_encoded]])

            # Prédiction
            result = int(xgb_model.predict(X)[0])

            # Mapping label binaire
            insurance_label = {0: "No Insurance", 1: "Insurance"}
            label = insurance_label.get(result, "Unknown")

            # Confiance (%)
            if hasattr(xgb_model, 'predict_proba'):
                probs = xgb_model.predict_proba(X)[0]
                confidence = round(float(np.max(probs) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction XGBoost")
            result, label, confidence = None, None, None

    return render(request, 'xgboost.html', {
        'form': form,
        'result': result,
        'label': label,
        'confidence': confidence,
        'model_missing': model_missing,
    })

def home(request):
    return render(request, 'dashboard.html')


def dashboard(request):
    return render(request, 'dashboard.html')

def regression_view(request):
    return render(request, 'regression.html')


def kmeans_view(request):
    return render(request, 'kmeans.html')
