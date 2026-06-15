import joblib
import numpy as np

from sklearn.linear_model import (
    LogisticRegression
)

X_train = np.load(
    "data/processed/train_embeddings.npy"
)

y_train = np.load(
    "data/processed/train_labels.npy",
    allow_pickle=True
)

clf = LogisticRegression(
    max_iter=1000
)

clf.fit(
    X_train,
    y_train
)

joblib.dump(
    clf,
    "models/sentiment_classifier.pkl"
)

print(
    "Classifier Saved"
)