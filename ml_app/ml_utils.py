import os


def _load_model(filename):
    """Try to load a model using joblib first, then pickle as a fallback.

    Raises FileNotFoundError if the file does not exist, and re-raises
    the underlying loader exception if both attempts fail.
    """
    path = os.path.join(os.path.dirname(__file__), 'ml_models', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    # First try joblib (commonly used for scikit-learn pipelines)
    try:
        import joblib

        return joblib.load(path)
    except Exception as job_err:
        # Fallback to pickle
        try:
            import pickle

            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as p_err:
            # Provide both errors to help debugging
            raise RuntimeError(
                f"Failed to load model '{path}': joblib error: {job_err!r}; pickle error: {p_err!r}"
            ) from p_err
