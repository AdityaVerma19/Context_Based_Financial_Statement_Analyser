import joblib
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

X_test = np.load(
    "data/processed/test_embeddings.npy"
)

y_test = np.load(
    "data/processed/test_labels.npy",
    allow_pickle=True
)

clf = joblib.load(
    "models/sentiment_classifier.pkl"
)

preds = clf.predict(
    X_test
)

print(
    classification_report(
        y_test,
        preds
    )
)