import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji

# NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load models
tfidf_loaded = joblib.load('tfidf_vector.pkl')
model_loaded = joblib.load('xgboost_classifier.pkl')
label_encoder_loaded = joblib.load('lablel_encoder.pkl')

# Stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess_text(text):
    text = emoji.replace_emoji(text, replace='')
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"@\w+|#\w+|\d+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Sentiment styles
sentiment_style = {
    "Positive": ("😊 Positive", "Your sentence expresses joy and positivity.", "green"),
    "Negative": ("😞 Negative", "This sentence conveys a negative emotion.", "red"),
    "Mixed": ("😐 Mixed", "Your sentence shows both positive and negative tones.", "orange"),
    "Neutral": ("😶 Neutral", "The sentiment is balanced or neutral.", "gray")
}

# Set page config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# Custom background
st.markdown("""
    <style>
        body {
            background: linear-gradient(120deg, #fdfbfb 0%, #ebedee 100%);
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        .result-text {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #ffffff;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 style='text-align: center; color: #3f51b5;'>💬 AI Sentiment Analyzer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 18px;'>Understand the emotional tone of your text in seconds</p>", unsafe_allow_html=True)
st.markdown("---")

# Text input
user_input = st.text_area("📝 Enter your sentence:", height=150, placeholder="Type here...")

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to analyze.")
    else:
        clean_text = preprocess_text(user_input)
        transformed_text = tfidf_loaded.transform([clean_text])
        prediction_encoded = model_loaded.predict(transformed_text)
        prediction = label_encoder_loaded.inverse_transform(prediction_encoded)[0]

        sentiment_label, sentiment_desc, color = sentiment_style.get(prediction, ("Unknown", "", "black"))

        # Result display
        st.markdown(f"""
            <div class="result-text" style="color: {color};">
                Sentiment: {sentiment_label} - {sentiment_desc}
            </div>
        """, unsafe_allow_html=True)