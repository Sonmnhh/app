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



# In[8]:

# Define the exact paths for the tokenizer and model
tokenizer_path = r"C:\Users\ASUS\Desktop\k2 2023-2024\Các hệ thống thông tin nâng cao\Final\tokenizer.pkl"
model_path = r"C:\Users\ASUS\Desktop\k2 2023-2024\Các hệ thống thông tin nâng cao\Final\finbert_model"

# Load the tokenizer
try:
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
except FileNotFoundError:
    st.error(f"Error loading the tokenizer: No such file or directory: '{tokenizer_path}'")
    st.stop()

# Load the model
try:
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
except FileNotFoundError:
    st.error(f"Error loading the model: No such file or directory: '{model_path}'")
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


# Streamlit app interface
input_text = st.text_area("Enter text for sentiment analysis")
if st.button("Analyze"):
    if input_text:
        sentiment, score = predict_sentiment(input_text)
        st.write(f"Sentiment: {sentiment} (Confidence: {score:.2f})")
    else:
        st.write("Please enter some text for analysis.")


# In[ ]:




