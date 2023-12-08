import streamlit as st
import numpy as np

# -----------------------------------------
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from transformers import AutoTokenizer
from transformers import pipeline

# ++++++++++++++++++++++++++++
# Vader
sia = SentimentIntensityAnalyzer()

# Roberta
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)

# pipeline
sentiment_analyzer = pipeline('sentiment-analysis')

# ++++++++++++++++++++++++++++

@st.cache_data
def vader(text):
    try:
        #{'neg': 0.119, 'neu': 0.743, 'pos': 0.139, 'compound': 0.1027}
        return sia.polarity_scores(text)
    except:
        pass

@st.cache_data
def roberta(text):
    try:
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores
    except:
        pass

@st.cache_data
def pipeline(text):
    try:
        return sentiment_analyzer(text)
    except:
        return "None"
# -----------------------------------------
method = "Vader"
with st.sidebar:
    st.header("Choose a Method")
    method = st.selectbox('Methods', (
        "Vader",
        "Roberta pré-trained Model",
        "Roberta with pipeline"
    ))



if method == "Vader":
    st.header(method)
    title = st.text_area('Your Text')
    button = st.button(label="Predict")
    if button:
        res =vader(title)
        if res['compound'] < -0.1:
            st.write(f'Your Text are classified as Negative with {res["neg"]:.4f}% as accuracy')
        if res['compound'] > 0.5:
            st.write(f'Your Text are classified as Positive with {res["pos"]:.4f}% as accuracy')
        else:
            st.write(f'Your Text are classified as Neutre with {res["neu"]:.4}% as accuracy')
            
        
elif method == "Roberta pré-trained Model":
    st.header(method)
    title = st.text_area('Your Text')
    button = st.button(label="Predict")
    if button:
        x = roberta(title)
        res = np.argmax(x)
        if res == 0:
            st.write(f'Your Text are classified as Negative with {x[res]:.4f}% as accuracy')
        if res == 1 :
            st.write(f'Your Text are classified as Positive with {x[res]:.4f}% as accuracy')
        else:
            st.write(f'Your Text are classified as Neutre with {x[res]:.4f}% as accuracy')
else:
    st.header(method)
    title = st.text_area('Your Text')
    button = st.button(label="Predict")
    if button:
        st.write(f"Your Text are classified as {pipeline(title)[0]['label']} sense with {pipeline(title)[0]['score']:.4f}% of accuracy")

