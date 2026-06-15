import joblib
import torch

from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

encoder = AutoModel.from_pretrained(MODEL_NAME)

classifier = joblib.load(
    "models/sentiment_classifier.pkl"
)

def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = encoder(**inputs)

    emb = outputs.last_hidden_state[:, 0, :]

    return emb.numpy()

def predict_sentiment(text):

    emb = get_embedding(text)

    pred = classifier.predict(emb)[0]

    probs = classifier.predict_proba(emb)[0]

    confidence = max(probs)

    return {
        "label": pred,
        "score": round(confidence * 100, 2)
    }