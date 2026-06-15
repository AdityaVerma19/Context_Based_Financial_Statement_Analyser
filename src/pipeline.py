from .predict import (
    predict_sentiment
)

from .fingpt_explain import (
    explain_sentiment
)

def analyze(text):

    sentiment = predict_sentiment(
        text
    )

    result = explain_sentiment(
        text,
        sentiment["label"]
    )

    return {
        "sentiment":
            result["sentiment"],

        "confidence":
            sentiment["score"],

        "explanation":
            result["explanation"]
    }