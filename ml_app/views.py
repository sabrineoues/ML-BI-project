from django.shortcuts import render
from .forms import KNNSampleForm
from django.conf import settings
import joblib
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def _load_model(rel_path):
    """Load a model from the given relative path with fallbacks and sklearn alias patching."""
    base_path = os.path.join(settings.BASE_DIR, rel_path)
    alt_path = os.path.join(os.path.dirname(__file__), rel_path.split('ml_app' + os.sep, 1)[-1])
    try:
        return joblib.load(str(base_path))
    except FileNotFoundError:
        try:
            return joblib.load(str(alt_path))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find model at {base_path} or {alt_path}") from e
    except AttributeError as e:
        msg = str(e)
        if 'sklearn.metrics._dist_metrics' in msg and 'EuclideanDistance' in msg:
            try:
                import importlib
                dm = importlib.import_module('sklearn.metrics._dist_metrics')
                if hasattr(dm, 'EuclideanDistance32'):
                    setattr(dm, 'EuclideanDistance', getattr(dm, 'EuclideanDistance32'))
                elif hasattr(dm, 'EuclideanDistance64'):
                    setattr(dm, 'EuclideanDistance', getattr(dm, 'EuclideanDistance64'))
                return joblib.load(str(base_path))
            except Exception:
                raise
        raise


def knn_view(request):
    form = KNNSampleForm(request.POST or None)
    result = None
    label = None
    confidence = None

    # Charger la pipeline — gérer l'absence du modèle pour l'UI
    model_missing = False
    try:
        knn_pipeline = _load_model('ml_app/ml_models/knn.pkl')
    except FileNotFoundError:
        knn_pipeline = None
        model_missing = True

    if form.is_valid() and knn_pipeline is not None:
        try:
            # Préparer les données (ordre EXACT des features attendu par le pipeline)
            salary = form.cleaned_data['salary']
            company_name_freq = form.cleaned_data['company_name_freq']
            location_freq = form.cleaned_data['Location_freq']

            X = np.array([[salary, company_name_freq, location_freq]])

            # Prédiction
            result = knn_pipeline.predict(X)[0]

            # Mapping label (si nécessaire)
            joblevel_label = {0: 'Junior', 1: 'Medior', 2: 'Senior'}
            label = joblevel_label.get(int(result), str(result))

            # Confiance (%)
            if hasattr(knn_pipeline, 'predict_proba'):
                probs = knn_pipeline.predict_proba(X)[0]
                confidence = round(float(np.max(probs) * 100), 2)

        except Exception:
            logger.exception("Erreur lors de la prédiction KNN")
            result, label, confidence = None, None, None

    context = {
        'form': form,
        'result': result,
        'label': label,
        'confidence': confidence,
        'model_missing': model_missing,
    }

    return render(request, 'knn.html', context)


def home(request):
    return render(request, 'dashboard.html')


def dashboard(request):
    return render(request, 'dashboard.html')


def xgb_view(request):
    return render(request, 'xgboost.html')


def regression_view(request):
    return render(request, 'regression.html')


def kmeans_view(request):
    return render(request, 'kmeans.html')
