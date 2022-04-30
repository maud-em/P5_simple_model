# API 
# 'tags.csv' contains the possible tags to suggest
# 'multitag_model.pkl' is the pickled Machine Learning model 

import pandas as pd
import numpy as np
import streamlit as st
import requests
import joblib
     
my_tags = pd.read_csv('tags.csv')
tags = my_tags['0'].values

def suggest_tags(tags, sentence_vec):
    tag_sugg = np.array(tags)[sentence_vec.astype(int)==1]
    return tag_sugg

def request_prediction(data):
    #multitag_model = open('multitag_model.pkl','rb')
    multitag_model = open('tfidf_model.pkl','rb')
    clf = joblib.load(multitag_model)
    my_prediction = clf.predict(data)
    #tag_bool = my_prediction.toarray()==1
    #tag_sugg = tags[tag_bool[0]]
    tag_sugg = suggest_tags(tags, my_prediction)
    return tag_sugg


def main():

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
