import pandas as pd
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModel
)

MODEL_NAME = "ProsusAI/finbert"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

def get_embedding(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = model(**inputs)

        cls_embedding = (
            outputs.last_hidden_state[:,0,:]
        )

    return cls_embedding.squeeze().numpy()


def process_file(
        csv_path,
        embedding_path,
        label_path
):

    df = pd.read_csv(csv_path)

    embeddings = []

    for text in df["text"]:

        emb = get_embedding(text)

        embeddings.append(emb)

    embeddings = np.array(
        embeddings
    )

    np.save(
        embedding_path,
        embeddings
    )

    np.save(
        label_path,
        df["label"]
    )

    print(
        embedding_path,
        embeddings.shape
    )


process_file(
    "data/processed/train.csv",
    "data/processed/train_embeddings.npy",
    "data/processed/train_labels.npy"
)

process_file(
    "data/processed/test.csv",
    "data/processed/test_embeddings.npy",
    "data/processed/test_labels.npy"
)