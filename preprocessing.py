import pandas as pd
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
import re
import spacy
import ssl
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from IPython.display import clear_output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly

df = pd.read_json("E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\json\\2013-01-08.json")
df.replace({"aye" : 1, "no" : 0}, inplace = True)
edited_json = df["speech"].copy()
df["speech-without-stopwords"] = edited_json

def preprocess_speech_data(data, name):
    # Proprocessing the data
    data[name] = data[name].str.lower()
    # Code to remove the Hashtags from the text
    data[name] = data[name].apply(lambda x:re.sub(r'\B#\S+','',x))
    # Code to remove the links from the text
    data[name] = data[name].apply(lambda x:re.sub(r"http\S+", "", x))
    # Code to remove the Special characters from the text 
    data[name] = data[name].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Code to substitute the multiple spaces with single spaces
    data[name] = data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    # Code to remove all the single characters in the text
    data[name] = data[name].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', '', x))
    # Remove the twitter handlers
    # data[name] = data[name].apply(lambda x:re.sub('@[^\s]+','',x))
    
def remove_stopwords_tokenize(data, name):
      
    def getting(sen):
        example_sent = sen
        filtered_sentence = [] 
        stop_words = set(stopwords.words("english")) 
        word_tokens = word_tokenize(example_sent) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        return filtered_sentence
    # Using "getting(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting(i))
    data[name]=x

lemmatizer = WordNetLemmatizer()
def Lemmatization(data,name):
    def getting2(sen):
        
        example = sen
        output_sentence =[]
        word_tokens2 = word_tokenize(example)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens2]
        
        # Remove characters which have length less than 2  
        without_single_chr = [word for word in lemmatized_output if len(word) > 2]
        # Remove numbers
        cleaned_data_title = [word for word in without_single_chr if not word.isnumeric()]
        
        return cleaned_data_title
    # Using "getting2(sen)" function to append edited sentence to data
    x=[]
    for i in data[name].values:
        x.append(getting2(i))
    data[name]=x
    
def make_sentences(data,name):
    data[name]=data[name].apply(lambda x:' '.join([i+' ' for i in x]))
    # Removing double spaces if created
    data[name]=data[name].apply(lambda x:re.sub(r'\s+', ' ', x, flags=re.I))
    
# Using the preprocessing function to preprocess the hotel data
preprocess_speech_data(data,"speech_without_stopwords")
# Using tokenizer and removing the stopwords
remove_stopwords_tokenize(data,"speech_without_stopwords")
# Converting all the texts back to sentences
make_sentences(data,"speech_without_stopwords")

#Edits After Lemmatization
final_Edit = data["speech_without_stopwords"].copy()
data["post_lemmatization"] = final_Edit

# Using the Lemmatization function to lemmatize the hotel data
Lemmatization(data,"post_lemmatization")
# Converting all the texts back to sentences
make_sentences(data,"post_lemmatization")

data.head(6)

