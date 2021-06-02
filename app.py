import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import re

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC


from stop_words import get_stop_words 

st.title('Labled data app')


# data = pd.read_csv('./data/labeled_data.csv')
# # data['tweet'] = data['tweet'].apply(lambda tweet: re.sub('[^a-zA-Z#]+', ' ', tweet.lower())) 
# # data['clean_Tweets'] = np.vectorize(remove_pattern)(data['tweet'], "[^A-Za-z]")
# data['tweet'] = data['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))
# data['tweet'] = data['tweet'].replace('[\!,]', '', regex=True)

# clf = make_pipeline(
#     TfidfVectorizer(stop_words=get_stop_words('en')),
#     OneVsRestClassifier(SVC(kernel='linear', probability=True))
# )

# clf = clf.fit(X=data['tweet'], y=data['class'])
# text = "I hate you, please die!"
# clf.predict_proba([text])[0]

# clf = make_pipeline(
#     TfidfVectorizer(stop_words=get_stop_words('en')),
#     OneVsRestClassifier(SVC(kernel='linear', probability=True))
# )
# clf = clf.fit(data,data.tweet)
#clf.predict_proba("i hate you")
title = st.text_input("label goes here")
clf = joblib.load('./data/filename.joblib') 
st.write(pd.DataFrame(clf.predict_proba([title]),columns=['Hate', 'Offensive', 'Neutral']))


