import pandas as pd


def load_phrasebank(filepath):

    texts = []
    labels = []

    with open(filepath,
              "r",
              encoding="latin1") as f:

        for line in f:

            line = line.strip()

            if "@positive" in line:
                text = line.replace("@positive", "")
                label = "positive"

            elif "@negative" in line:
                text = line.replace("@negative", "")
                label = "negative"

            elif "@neutral" in line:
                text = line.replace("@neutral", "")
                label = "neutral"

            else:
                continue

            texts.append(text)
            labels.append(label)

    return pd.DataFrame({
        "text": texts,
        "label": labels
    })