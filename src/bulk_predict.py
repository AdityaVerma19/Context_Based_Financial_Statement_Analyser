import pandas as pd
from .predict import predict_sentiment

df = pd.read_csv(
    "data/raw/influencer_statements.csv"
)

results = []

for _, row in df.iterrows():

    pred = predict_sentiment(
        row["text"]
    )

    results.append({
        "person": row["person"],
        "statement": row["text"],
        "sentiment": pred["label"],
        "confidence": pred["score"]
    })

pd.DataFrame(results).to_csv(
    "outputs/predictions.csv",
    index=False
)

print("Done")