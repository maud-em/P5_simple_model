import unicodedata
from nltk.stem import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
#from bs4 import BeautifulSoup
import spacy
nlp = spacy.load('en_core_web_sm')

stop_words = stopwords.words("english")
useless_words = ['way', 'difference', 'value', 'use', 'method', 'code', 'view', 'test', 'work', 'page',
                 'problem', 'question', 'solution', 'thanks', 'call', 'line', 'thing', 'issue',
                 'change','result', 'idea', 'edit', 'name', 'number','help','error','need','answer','event',
                 'column','file','case', 'element']
stop_words = stop_words + useless_words

#BeautifulSoup(x).get_text()

class textNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.stopwords  = set(stop_words)
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = nlp

    def is_punct(self, token):
        return all(
            unicodedata.category(char).startswith('P') for char in token
        )

    def is_stopword(self, token):
        return token.lower() in self.stopwords
    
    def transform_token(self, token):
        token = self.transform_string(token)
        token = self.domain_specific_words(token)
        token = self.remove_version_digits(token)
        if not self.is_punct(token) and not self.is_stopword(token):
            return self.lemmatizer.lemmatize(token)
        else: 
            return ''
 
    
    def normalize(self, document):
        normalized_doc = [self.transform_token(token.text) for token in nlp(document) if token.pos_ == "NOUN" or token.pos_ == "PROPN"]
        return ' '.join(normalized_doc)
       
    
    def transform_string(self,token):
        # Make lower
        token = token.lower()
        # Contractions
        token = re.sub(r"can't", r"can not", token)
        token = re.sub(r"i'm", r"i am", token)
        token = re.sub(r"'s", r" is", token)
        token = re.sub(r"n't", r" not", token)
        token = re.sub(r"'ve", r" have", token)
        token = re.sub(r"'ll", r" will", token)
        token = re.sub(r"'d", r" would", token)    
        return token

    
    def domain_specific_words(self, token):
       # Code specific words
        token = re.sub(r'amazon web service', 'aws', token)
        token = re.sub(r'amazon-web-service', 'aws', token)
        token = re.sub(r'objective-c', 'objectivec', token)
        token = re.sub(r'objective c', 'objectivec', token)
        token = re.sub(r'c\+\+|c \+\+', 'cplusplus', token)
        token = re.sub(r'c#|c #', 'csharp', token)
        token = re.sub(r'g\+\+|c \+\+', 'gplusplus', token)
        token = re.sub(r'g#|g #', 'gsharp', token)
        token = re.sub(r'(\b)r(\b)', ' languager ', token)
        token = re.sub(r'(\b)c(\b)', ' languagec ', token)
        token = re.sub(r'.net', 'dotnet', token)
        token = re.sub(r'.js', 'dotjs', token)
        return token
    
    def remove_version_digits(self,token):
        # Remove version numbers of type ...x.x.
        token = re.sub(r'.x', '', token)
        token = re.sub('(?:-[\d\.]+)*', '', token)
        # Remove digits    
        token = re.sub(r'\d', '', token)
        return token
 

    def filter_nouns(self, token):
        if token.pos_ == "NOUN" or token.pos_ == "PROPN":
            return token

    def fit(self, X, y=None):
        return self

    def transform(self, doc_df):
        return pd.Series([self.normalize(document) for document in doc_df])
