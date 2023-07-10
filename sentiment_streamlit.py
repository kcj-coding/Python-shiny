import pandas as pd
import numpy as np
import html
import re
import keras
from keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from keras_preprocessing.sequence import pad_sequences
import joblib
import string
import streamlit as st

# set app configuration ####
st.title("Sentiment Analysis App")
st.markdown("Text sentiment")
text = st.text_area("Enter text")
#st.checkbox("Show tokens")

############# clean text data #####################
def clean(text):
    # convert html escapes to characters
    text = html.unescape(text)
    # tags like <tab>
    text = re.sub(r'<[^<>]*>', ' ', text)
    # markdown URLs
    text = re.sub(r'\[([^\[\]]*)\]\([^\(\)]*\)', r'\1', text)
    # text in code or brackets
    text = re.sub(r'\[[^\[\]]*\]', ' ', text)
    # standalone sequences of specials
    text = re.sub(r'(?:^|\s)[&#<>{}\[\]+|\\:-]{1,}(?:\s|$)', ' ', text)
    # standalone sequences of hyphens
    text = re.sub(r'(?:^|\s)[\-=\+]{2,}(?:\s|$)', ' ', text)
    # sequences of whitespaces
    text = re.sub(r'\s+', ' ', text)
    # lowercase
    text = text.lower()
    # remove punctuation
    text = text.strip(string.punctuation)
    
    return text.strip()

# make as df

texter = text

df = pd.DataFrame({'data':[texter]})

df1 = df['data'].apply(clean)

### load and config for ml model #####
def run_ml_model():
    ml_vectorizer = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\vectorizer.pkl")
    
    texter = ml_vectorizer.transform(df1)
    
    ml_model = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\svc_model.pkl")
    
    preds = ml_model.predict(texter)
    prob = np.max(ml_model.predict_proba(texter), axis=1)
    
    st.success("machine learning model says: "+str(preds)+" with probability: "+str(prob))
    
    # transform text data
    
    # predict new values
    

### load and config for dl model #####
def dl_model_chk(value):
    if value < 0.5:
        return "Negative"
    else:
        return "Positive"
    
def results_chk(value):
    if value < 0.5:
        return 1-value
    else:
        return value


def run_dl_model():
    
    token = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\token.pkl")
    
    dl_model = load_model("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\cnn_dl.hdf5")
    
    test_data = token.texts_to_sequences(df1)
    
    test_data = pad_sequences(test_data, padding='post', maxlen=200)
    
    preds1 = dl_model.predict(test_data)
    
    preds2 = preds1#.astype(float)
    
    #preds2 = [1 - abs(x-0.5) for x in preds2]
    
    results = pd.DataFrame({'results':[preds1], 'results_proba':[preds2]})
    
    results1 = pd.DataFrame({'results_proba':[preds2[:, 0]]})
    
    results1['results_proba'] = results1['results_proba'].apply(results_chk)
    
    results['results'] = results['results'].apply(dl_model_chk)
    
    #results1['results_proba'] = 1-abs((results1['results_proba'])-0.5)
    
    #df11 = df
    
    #df11['preds'] = results
    #df11['preds_proba'] = 1-abs(results-0.5)
    
    #df11['preds'] = df11['preds'].apply(dl_model_chk)
        
    st.success("deep learning model says: "+str(results['results'].values)+" with probability: "+str(results1['results_proba'][0]))
    
def search(mydict,value):
    for key in mydict.items():
        if value in key:
            st.write(key)

def get_tokens():
    
    token = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\token.pkl")
    
    test_data = token.texts_to_sequences(df1)
    
    t = token.word_index
    
    tt = df['data'][0].split()
    
    for wordies in tt:
        search(t,wordies)
    
    #st.write(test_data)
    st.success("there are "+str(str(len(text.split())))+" words, but only "+str(len(test_data[0]))+" tokens")

#### run conditions
if st.checkbox("Show tokens"):
    get_tokens()
    
if st.button("Submit"):
    run_ml_model()
    run_dl_model()
    
