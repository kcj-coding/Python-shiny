from shiny import *
from shiny import App, reactive, render, ui
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


########## app config ###################

app_ui = ui.page_fluid(
    ui.page_navbar(title="Sentiment Analysis",bg="#42b3f5", inverse=True),
    ui.h2("Sentiment analysis"),
    ui.panel_sidebar(
    ui.input_text_area("txt", "Enter the text to display below:"),
    ui.input_checkbox("token_btn", "Show tokens", False),
    ui.input_action_button(
                    "submit_btn", "Submit", class_="btn-primary w-100")),
    #ui.row(
    #    #ui.column(6, ui.output_text("text")),
    #    ui.column(6, ui.output_text_verbatim("token_output", placeholder=True))
    #),
    ui.panel_main(
    ui.row(
        ui.column(10, ui.output_text_verbatim("indices", placeholder=False))),
    ui.row(
        ui.column(10, ui.output_text_verbatim("tokens", placeholder=False))),
    ui.row(
        ui.column(10, ui.output_text_verbatim("dl_output", placeholder=False)))
    ),
    title="Sentiment App"
)


# config #################

def server(input: Inputs, output: Outputs, session: Session):
    
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

        texter = input.txt()

        df = pd.DataFrame({'data':[texter]})
        
        df1 = df['data'].apply(clean)
        
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
            
        return "deep learning model says: "+str(results['results'].values)+" with probability: "+str(results1['results_proba'][0])
    

    def get_indices():
        
        texter = input.txt()

        df = pd.DataFrame({'data':[texter]})
        
        df1 = df['data'].apply(clean)
        
        token = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\token.pkl")
        
        test_data = token.texts_to_sequences(df1)
        
        t = token.word_index
        
        tt = df['data'][0].split()
        
        dfer = []
        for wordies in tt:
            for key in t.items():
                if wordies in key:
                    dfer.append(key)
        
        return dfer
    
    #def search(mydict,value):
    #    global dfer
    #    dfer = []
    #    for key in mydict.items():
    #        if value in key:
    #            dfer.append(key)

    def get_tokens():
        
        texter = input.txt()

        df = pd.DataFrame({'data':[texter]})
        
        df1 = df['data'].apply(clean)
        
        token = joblib.load("C:\\Users\\kelvi\\Desktop\\sentiment analysis\\python models\\token.pkl")
        
        test_data = token.texts_to_sequences(df1)
        
        t = token.word_index
        
        tt = df['data'][0].split()
        
        #for wordies in tt:
        #    search(t,wordies)
        
        dfer = []
        for wordies in tt:
            for key in t.items():
                if wordies in key:
                    dfer.append(key)
                    
        dfer = pd.DataFrame({'data1':[dfer]})
        
        return str(dfer['data1'].values)  +"\n" + "there are "+str(str(len(texter.split())))+" words, but only "+str(len(test_data[0]))+" tokens"
    
    @output
    @render.text
    #@reactive.event(input.token_btn, ignore_none=True)
    
    #def indices(): 
    #    if input.token_btn():
    #       return get_indices()

    
    def tokens(): 
        if input.token_btn():
           return get_tokens()
    
    @output
    @render.text
    @reactive.event(input.submit_btn, ignore_none=True)
    
    
    
    def dl_output():
            return run_dl_model()


app = App(app_ui, server)






    
    
