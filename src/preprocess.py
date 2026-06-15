from sklearn.model_selection import train_test_split
from .data_loader import load_phrasebank

df = load_phrasebank(
    "data/raw/Sentences_AllAgree.txt"
)

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

train_df.to_csv(
    "data/processed/train.csv",
    index=False
)

test_df.to_csv(
    "data/processed/test.csv",
    index=False
)

print("Done")