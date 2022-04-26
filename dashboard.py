import pandas as pd
import numpy as np
import streamlit as st
import requests
import joblib
   
my_tags = pd.read_csv('tags.csv')
tags = my_tags['0'].values


def request_prediction(data):
    multitag_model = open('multitag_model.pkl','rb')
    clf = joblib.load(multitag_model)
    my_prediction = clf.predict(data)
    tag_bool = my_prediction.toarray()==1
    tag_sugg = tags[tag_bool[0]]
    return tag_sugg


def main():
    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'
    FLASK_URI = 'http://127.0.0.1:5000/'
    #CORTEX_URI = 'http://0.0.0.0:8890/'
    #RAY_SERVE_URI = 'http://127.0.0.1:8000/regressor'

    #api_choice = st.sidebar.selectbox(
    #    'Quelle API souhaitez vous utiliser',
    #    ['MLflow', 'Cortex', 'Ray Serve'])

    st.title('Tag suggestion to StackOverflow questions')

    title = st.text_input('Title of your StackOverflow question','')

    body = st.text_area('Ask your question','')

    predict_btn = st.button('Tag Suggestion')
    if predict_btn:
        data = [title]
        #data=title + ' ' + body
        pred = request_prediction(data)
        
        st.write(pred)


if __name__ == '__main__':
    main()
