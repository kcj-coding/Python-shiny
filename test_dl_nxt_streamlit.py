import pandas as pd
import os
import numpy as np
import re
import html
import tensorflow as tf
import pickle
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
import streamlit as st

# set app configuration ####
st.title("Text Generation App")
st.markdown("Text creation")
text = st.text_area("Enter text")
temper = st.slider('Temperature', 0.01, 2.00, 0.01)
word_numer = st.slider('Words to generate', 1, 8, 1)
orderr = st.checkbox("Randomise word likelihood")
wordrr = st.checkbox("Randomise words")
words_to_skip = st.checkbox("Do not repeat words")

# credit to
# https://www.kaggle.com/code/ysthehurricane/next-word-prediction-bi-lstm-tutorial-easy-way
# https://medium.com/@david.campion/text-generation-using-bidirectional-lstm-and-doc2vec-models-1-3-8979eb65cb3a

def run_dl_model():

    # load the tokenizer
    tokenizer = pickle.load(open("tokenizer.pkl","rb"))
    
    # load the model
    model = tf.keras.models.load_model("next_word_bi-lstm.h5")#pickle.load(open("next_word_bi-lstm.h5","rb"))
    
    # load max sequence len
    max_sequence_len = pickle.load(open("max_sequence.pkl","rb"))
    
    total_words = len(tokenizer.word_index) + 1
    seed_text1 = text
    temp = temper
    last_word = seed_text1.split()
    last_word = last_word[-1]
    seed_text = seed_text1
    outputs = ""
    scores = ""
    next_words = word_numer + 1
    
    def sample(preds, temperature=1.0):
        #global probas
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = preds#np.random.multinomial(1, preds[:, 0], size=1)
        return probas#np.argmax(probas)
    
    num_prv = 0
    #words_predicted = pd.DataFrame({'name':[0]})
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    #     predicted = model.predict_classes(token_list, verbose=0)
        predicted = model.predict(token_list)
        predicted = sample(predicted,temperature=temp)
        # np.fliplr can be used to rearrange order descending
        top_values_probs = np.sort(predicted, axis=1)[:,-5:]#predicted#.numpy()
        if orderr:
            top_values_probs = np.random.random_sample((5,))
            top_values_probs = pd.Series(top_values_probs)
            #top_values_probs = top_values_probs.drop(labels=[0])
        #top_values = [predicted[i] for i in np.argsort(predicted,axis=1)]#[-5:]]
        #prediction_probabilities = tf.math.top_k(predicted, k=5)
        prediction_probs = np.argsort(predicted, axis=1) [:,-5:]
        top_values = np.argpartition(predicted,-5)[-5:] #np.argsort(predicted, axis=1) [:,-5:]
        if wordrr:
            top_values = np.random.randint(1, total_words, size=5)
            top_values = pd.Series(top_values)
        probabilities = np.argpartition(predicted,-5)[-5:]
        
        # get scores and word tokens
        if orderr:
            top_values_probs = pd.DataFrame(top_values_probs)#.T # transpose
        else:
            top_values_probs = pd.DataFrame(top_values_probs).T # transpose
        top_values_probs = top_values_probs.rename(columns={0:'pred'})
        top_values_probs = round(top_values_probs['pred'],3)[1:5]
        if wordrr:
           top_values = pd.DataFrame(top_values)#.T # transpose 
        else:
            top_values = pd.DataFrame(top_values).T # transpose
        top_values = top_values.rename(columns={0:'name'})
        top_values = top_values['name'][1:5]
        
        top_values = pd.concat([top_values,top_values_probs],axis=1)
        
        if words_to_skip == True:
            already_used = pd.DataFrame(token_list).T
            already_used = already_used.rename(columns={0:'name'})
            #words_predicted = pd.concat([words_predicted,already_used],axis=0)#, ignore_index=True)
            top_values2 = top_values.name.isin(already_used.name)
            top_values2 = pd.DataFrame(top_values2)
            top_values2 = top_values2.rename(columns={'name':'exists'})
            top_values1 = pd.concat([top_values,top_values2], axis=1)
            top_values3 = top_values[top_values1['exists']==False]
            top_values4 = top_values3[['name','pred']]
            top_values = top_values4
            
            # check if is empty if so error
            
        
        
        predicted=np.argmax(predicted,axis=1)
        output_word = ""
        
        num_prv = _ + 1
        counterer = 0
        for word, index in tokenizer.word_index.items():
            #range(0,len(top_values)):
            #print(index)
            for i in top_values['name']:
                
                if index == i:#top_values[i]:
                    counterer = counterer + 1
                   # try:
                    #locale = top_values[top_values['name'] == counterer].index
                    tester1 = top_values['pred'].iloc[counterer-1]#.values
                    #tester1 = tester1.iloc[counterer]
                    scores += "," + str(tester1)
                    #except:
                    #    pass
                    if tester1.isna:
                        output_word = "error - no more words"
                    else:
                        output_word = word
                    outputs += "," + output_word  + str(tester1) + str(_+1)
    
                    #break
        t = str.split(outputs,",")
        t = list(filter(None, t))
        scores1 = str.split(scores,",")
        scores1 = list(filter(None, scores1))
        seed_text += " " + output_word
    #print(seed_text)
    #print("next possible words: ",t)
    
    # make df of new options
    tt = pd.DataFrame(t)
    tt = tt.rename(columns={0:"value"})
    #tt['num'] = tt.value.str.extract('(\d+)') # extract number
    tt['num'] = tt.value.str[-1] # get last letter
    #tt['word'] = tt.value.str.replace('\d+', '') # replace number with nothing
    tt['word'] = tt.value.str[:-1] # all except last letter
    tt['pred_num'] = tt.value.str.extract('(\d+\.\d+)') # extract prediction number (float)
    #tt['pred_num'] = tt['pred_num'].astype(str)
    #tt['pred_num'] = tt.pred_num.str[:-1] # all except last number
    #if orderr:
    #    if int(_) % 2:
    #        tt = tt.sort_values(by=['num','pred_num'], ascending = False)
    #    else:
    #        tt = tt.sort_values(by=['num','pred_num'], ascending = True)
    #else:
    tt = tt.sort_values(by=['num','pred_num'], ascending = False)
    #tt = tt.iloc[::-1]
    
    # list next possible words by number grouping
    
    
    
    #worders = []
    #for each in 
    tfer = []
    for i in range(0,next_words):
        tt1 = tt[tt['num'] == str((i+1))]
        dfer = tt1['word'].to_list()
        tfer.append(dfer)
    
    tfer = pd.DataFrame({'Value':tfer})
    tfer['word_num'] = range(1, len(tfer) + 1)
    
    
    stringer = ""
    seed_text2 = seed_text1
    for eacher in range(1,next_words+1):
        # new word
        tfer1 = tfer[tfer['word_num'] == eacher]
        tfer1 = tfer1['Value'].values
        tfer1 = tfer1[0] # to select only the words
        
        # prv word
        try:
            tfer10 = tfer[tfer['word_num'] == eacher-1]
            tfer10 = tfer10['Value'].values
            tfer10 = tfer10[0] # to select only the words
            stringer += ",,"+str(list(tfer10)[0])
            stringer = stringer.split(",,")
            stringer = stringer[1]
            # just get word (no number)
            stringer = ''.join(i for i in stringer if not i.isdigit()) # replace number with nothing
            stringer = stringer[:-1]
        except:
            pass#stringer = ""
            
        if eacher == 1:
            #st.success('---')
            st.success('your phrase: '+ seed_text2)
            st.success('for word: [' + last_word + "], " + "next predictions:" + " " + str(tfer1))
            seed_text2 = seed_text2 + " " + stringer
        else:
            seed_text2 = seed_text2 + " " + stringer
            last_worder = seed_text2.split()
            last_worder = last_worder[-1]
            # just get word (no number)
            last_worder = ''.join(i for i in last_worder if not i.isdigit()) # replace number with nothing
            #last_worder = last_worder[:-1]
            #st.success('---')
            
            if not tfer1:
                # only invoked if words_to_skip is checked/true
                st.success("Error - no more words available to predict that have not already been used")
            else:
                st.success('the phrase: '+ seed_text2)
                st.success('for word: [' + str(last_worder) + "], " + "next predictions:" + " " + str(tfer1))
        
#### run conditions
   
if st.button("Submit"):
    run_dl_model()
    
# streamlit run test_dl_nxt_streamlit.py