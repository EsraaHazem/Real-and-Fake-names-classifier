# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import keras
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
import numpy as np


df = pd.read_csv("C:/Users/Esraa/Desktop/Digified/final.csv")
df.drop(['index','Unnamed: 0'], axis=1, inplace=True)
df.drop_duplicates(inplace = True)
tags = df['class']
encoder = LabelEncoder()
encoder.fit(tags)
def f1_metric(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  recall = true_positives / (possible_positives + K.epsilon())
  f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
  return f1_val
dependencies = {
    'f1_metric': f1_metric
}

model = keras.models.load_model('C:/Users/Esraa/Desktop/Digified/my_model.h5', custom_objects=dependencies)
# loading
with open('C:/Users/Esraa/Desktop/Digified/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
def prediction(name):
  tokens = tokenizer.texts_to_matrix([name], mode='tfidf')
  predict_x=model.predict(np.array(tokens)) 
  c=np.argmax(predict_x,axis=1)
  xc = encoder.inverse_transform(c)
  if predict_x[0][1]> 0.50 :
    if predict_x[0][1]- predict_x[0][0]>0.001:
      Confidence="True Name with High Confidence"
    else:
      Confidence="True Name with Low Confidence"
  else:
    Confidence="False Name"
  return xc[0],Confidence

def main():
    st.set_page_config(page_title='real and fake names',page_icon=":tada:")
    st.title("REAL and FAKE NAMES WEB APP")
    name=st.text_input("Name")
    test="not find"
    Confidence="not find"

    if st.button('Result'):
        test,Confidence=prediction(name)

    st.title("first mode")
    st.success(test)
    st.success(Confidence)



if __name__ =='__main__':
    main()
    

