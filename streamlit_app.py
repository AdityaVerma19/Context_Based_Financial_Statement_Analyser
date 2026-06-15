import streamlit as st
from src.pipeline import analyze

# Set Page Config
st.set_page_config(
    page_title="Financial Sentiment Analyzer & Explainer",
    page_icon="📊",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .sentiment-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 24px;
        margin-bottom: 20px;
    }
    .positive-box {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 2px solid #a5d6a7;
    }
    .negative-box {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #ef9a9a;
    }
    .neutral-box {
        background-color: #efebe9;
        color: #4e342e;
        border: 2px solid #d7ccc8;
    }
    </style>
""", unsafe_allow_html=True)

# App Title & Header
st.title("📊 Context-Based Financial Statement Analyzer")
st.markdown("""
    This interactive tool extracts financial sentiment from statement inputs using a hybrid **FinBERT embedding** and **Logistic Regression** classifier, then applies **Qwen** reasoning to resolve keyword-level biases and explain the outputs.
""")

st.markdown("---")

# Sidebar Details
st.sidebar.header("🛠️ Pipeline Details")
st.sidebar.markdown("""
- **Embeddings**: FinBERT (`ProsusAI/finbert`) base layers extracting `[CLS]` token vectors.
- **Classifier**: Custom Logistic Regression head trained on the Financial Phrasebank.
- **Reasoning Layer**: Qwen (`Qwen/Qwen3-0.6B`) checking classifier alignment, executing label overrides, and generating insights.
""")

# Example Statements
st.sidebar.subheader("💡 Example Statements")
examples = [
    "Production costs have declined across several facilities.",
    "AI chip demand continues to exceed supply.",
    "Inflation continues to burden American consumers.",
    "The company will hold its annual general meeting on Thursday."
]

selected_example = st.sidebar.selectbox("Click to auto-fill an example:", ["-- Select Example --"] + examples)

# Input Block
input_text = ""
if selected_example != "-- Select Example --":
    input_text = selected_example

statement = st.text_area(
    "Enter Financial Statement:",
    value=input_text,
    placeholder="Type a financial news headline or corporate statement here..."
)

# Run Inference
if st.button("Analyze Sentiment", type="primary"):
    if statement.strip() == "":
        st.warning("Please enter a valid statement to analyze.")
    else:
        with st.spinner("Processing through FinBERT and Qwen Reasoning layers..."):
            try:
                # Call pipeline analyze
                result = analyze(statement)
                
                sentiment = result["sentiment"].lower()
                confidence = result["confidence"]
                explanation = result["explanation"]
                
                st.markdown("### 📈 Analysis Results")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Predicted Sentiment:**")
                    if sentiment == "positive":
                        st.markdown('<div class="sentiment-box positive-box">🟢 Positive</div>', unsafe_allow_html=True)
                    elif sentiment == "negative":
                        st.markdown('<div class="sentiment-box negative-box">🔴 Negative</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="sentiment-box neutral-box">⚪ Neutral</div>', unsafe_allow_html=True)
                        
                    st.metric(label="Classifier Confidence Score", value=f"{confidence}%")
                    st.progress(confidence / 100.0)
                    
                with col2:
                    st.markdown("**Reasoning & Insight Explanation:**")
                    st.info(explanation)
                    
            except Exception as e:
                st.error(f"Error executing pipeline: {e}")
                st.warning("Make sure your models are loaded and GPU/CPU configuration is correct.")
