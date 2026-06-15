import re
from transformers import pipeline

generator = pipeline(
    "text-generation",
   # model="HuggingFaceTB/SmolLM2-1.7B",
   #model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    model="Qwen/Qwen3-0.6B",
    device_map="auto"
)

def explain_sentiment(statement, sentiment):

    prompt = f"""You are a financial analyst.

Statement: The company is facing declining sales and rising debt.
Classifier Predicted Sentiment: negative
True Sentiment: negative
Reason: Declining sales combined with rising debt indicate deteriorating financial health and liquidity issues.

Statement: Net income rose by twenty percent this year.
Classifier Predicted Sentiment: positive
True Sentiment: positive
Reason: A substantial rise in net income demonstrates strong profitability and operational growth.

Statement: Production costs have declined across several facilities.
Classifier Predicted Sentiment: negative
True Sentiment: positive
Reason: Declined production costs mean lower expenses, which directly improves gross margins and profitability.

Statement: The company will hold its annual general meeting on Thursday.
Classifier Predicted Sentiment: neutral
True Sentiment: neutral
Reason: Announcing the date of an annual meeting is a standard administrative event with no direct sentiment impact.

Statement: {statement}
Classifier Predicted Sentiment: {sentiment}
True Sentiment:"""

    output = generator(
        prompt,
        max_new_tokens=80,
        do_sample=False
    )

    text = output[0]["generated_text"]
    response_text = text.replace(prompt, "").strip()

    # Parse the True Sentiment and Reason from the completion
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    
    predicted_sentiment_corrected = sentiment
    explanation = response_text
    
    if lines:
        # Check first line for corrected sentiment
        first_line = lines[0].lower()
        if "positive" in first_line:
            predicted_sentiment_corrected = "positive"
        elif "negative" in first_line:
            predicted_sentiment_corrected = "negative"
        elif "neutral" in first_line:
            predicted_sentiment_corrected = "neutral"
            
        # Find the line containing the Reason
        reason_found = False
        for line in lines:
            if "reason:" in line.lower():
                explanation = re.sub(r'(?i)^reason:\s*', '', line).strip()
                reason_found = True
                break
        
        if not reason_found and len(lines) > 1:
            explanation = " ".join(lines[1:])
            
    return {
        "sentiment": predicted_sentiment_corrected,
        "explanation": explanation.strip()
    }