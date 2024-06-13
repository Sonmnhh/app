#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pickle
import torch
import scipy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import pickle
import requests
import tempfile


# In[8]:

# Define the URLs for the tokenizer and model
tokenizer_url = 'https://drive.google.com/file/d/1nM-nXY308A0CtEwAUCq_Mhv5kde6yRO0/view?usp=drive_link'
model_url = 'https://drive.google.com/file/d/1QiZrC53r_Dqn5160rGvnHGzpzTG0hbqm/view?usp=drive_link'

# Download the tokenizer
@st.cache_data
def download_tokenizer(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

tokenizer_path = download_tokenizer(tokenizer_url)

# Load the tokenizer
try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
except Exception as e:
    st.error(f"Error loading the tokenizer: {e}")
    st.stop()

# Download the model
@st.cache_data
def download_model(url):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(response.content)
        return tmp_file.name

model_path = download_model(model_url)

# Load the model
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()
# In[9]:


# Function to get prediction
def predict_sentiment(text):
    tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
        logits = model(**inputs).logits
        scores = {
            k: v for k, v in zip(
                model.config.id2label.values(),
                torch.softmax(logits, dim=1).numpy().squeeze()
            )
        }
        sentiment = max(scores, key=scores.get)
        return sentiment, scores[sentiment]


# In[10]:
# Sidebar navigation
st.sidebar.title("For Financial")
page = st.sidebar.radio("Go to", ["Home", "News", "Contact Me"])

# Add custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    h1 {
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    p {
        color: #666666;
        text-align: center;
        font-size: 18px;
    }
    input {
        width: 100%;
        padding: 15px;
        margin: 10px 0;
        box-sizing: border-box;
        border: 2px solid #ccc;
        border-radius: 4px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 15px 20px;
        margin: 8px 0;
        border: none;
        border-radius: 4px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .decorative {
        background-image: url('https://www.transparenttextures.com/patterns/white-waves.png');
        padding: 50px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Financial Sentiment Analysis WebApp.")  
st.write('Welcome to my sentiment analysis app!')

# Streamlit app interface
input_text = st.text_area("Enter text for sentiment analysis")
if st.button("Analyze"):
    if input_text:
        sentiment, score = predict_sentiment(input_text)
        st.write(f"Sentiment: {sentiment} (Confidence: {score:.2f})")
    else:
        st.write("Please enter some text for analysis.")


# In[ ]:




