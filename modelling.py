import pandas as pd
import numpy as np
import nltk
import json
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
from nltk.tokenize import word_tokenize 
import re
import spacy
import ssl

from IPython.display import clear_output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly   

_JSON_INPUT_FOLDER = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\json"
_JSON_TEST_FILE = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\json\\2013-01-08.json"

## Preprocessing

with open(_JSON_TEST_FILE, "r") as file:
    df = pd.read_json(file.read())

df.replace({"aye" : 1, "no" : 0}, inplace = True)

edited_speech = df["text"].copy()
df["speech_without_stopwords"] = edited_speech

def preprocess_speech_data(data, column):
    # Convert text to lowercase
    data[column] = data[column].astype(str).str.lower()
    # Remove "{'p': ' ... '} from each sentence
    data[column] = data[column].apply(lambda x:re.sub(r'^.{5}|.{2}$', "", x))
    # Remove any links
    data[column] = data[column].apply(lambda x:re.sub(r"http\S+", "", x))
    # Remove all special characters
    data[column] = data[column].apply(lambda x:' '.join(re.findall(r'\w+', x)))
    # Replace multiple spaces with single spaces
    data[column] = data[column].apply(lambda x:re.sub(r'\s+', " ", x, flags=re.I))
    # Remove all single characters in the text
    data[column] = data[column].apply(lambda x:re.sub(r'\s+[a-zA-Z]\s+', "", x))  
    
# preprocess_speech_data(df, "speech_without_stopwords")
# print(df.head(5))
    
def remove_stopwords_tokenize(data, column):
    
    nltk.download("stopwords", quiet = True)
    nltk.download("punkt", quiet = True)
      
    def getting_filter(paragraph):
        example_paragraph = paragraph
        filtered_paragraph = [] 
        stop_words = set(stopwords.words("english")) 
        word_tokens_filter = word_tokenize(paragraph) 
        filtered_paragraph = [w for w in word_tokens_filter if not w in stop_words] 
        return filtered_paragraph
    
    # Append filtered paragraph to data
    x = []
    for i in data[column].values:
        x.append(getting_filter(i))
    data[column] = x
    
# remove_stopwords_tokenize(df, "speech_without_stopwords")
# print(df.head(5))

lemmatizer = WordNetLemmatizer()

def lemmatize_data(data, column):
    
    def getting_lemmatize(paragraph):
        
        nltk.download("wordnet", quiet = True)
        
        example_paragraph = paragraph
        output_paragraph =[]
        word_tokens_lemmatize = word_tokenize(paragraph)
        lemmatized_output = [lemmatizer.lemmatize(w) for w in word_tokens_lemmatize]
        
        # Remove characters where length < 2
        without_single_char = [word for word in lemmatized_output if len(word) > 2]
        # Remove numbers
        cleaned_data = [word for word in without_single_char if not word.isnumeric()]
        
        return cleaned_data
    
    # Append lemmatized paragraph to data
    x = []
    for i in data[column].values:
        x.append(getting_lemmatize(i))
    data[column] = x
    
# lemmatize_data(df, "speech_without_stopwords")
# print(df.head(5))

def join_paragraph(data, column):
    data[column] = data[column].apply(lambda x:" ".join([i + " " for i in x]))
    # Remove any double spaces
    data[column] = data[column].apply(lambda x:re.sub(r'\s+', " ", x, flags = re.I))
    
preprocess_speech_data(df, "speech_without_stopwords")
remove_stopwords_tokenize(df, "speech_without_stopwords")
join_paragraph(df, "speech_without_stopwords")

final_data = df["speech_without_stopwords"].copy()
df["post_lemmatization"] = final_data

lemmatize_data(df, "post_lemmatization")
join_paragraph(df, "post_lemmatization")

pos = neg = obj = count = 0

pos_tagging = []

# Append POS (part of speech) tag to individual words

nltk.download("averaged_perceptron_tagger", quiet = True)

for speech in df["post_lemmatization"]:
    list = word_tokenize(speech)
    pos_tagging.append(nltk.pos_tag(list))

df["pos_tags"] = pos_tagging

def tags_to_wordnet(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("R"):
        return wordnet.ADV
    elif tag.startswith("V"):
        return wordnet.VERB
    return None

# Function that returns list of pos-neg and objective score
# Returns empty list if not present in SentiWordNet
def get_sentiment(word, tag):
    
    nltk.download("sentiwordnet", quiet = True)
    
    wordnet_tag = tags_to_wordnet(tag)
    
    if wordnet_tag not in (wordnet.NOUN, wordnet.ADJ, wordnet.ADV):
        return []

    # Lemmatize words
    lemma = lemmatizer.lemmatize(word, pos = wordnet_tag)
    if not lemma:
        return []

    # Look up word using Synsets (synonymous words), part of WordNet
    synsets = wordnet.synsets(word, pos = wordnet_tag)
    if not synsets:
        return []

    synset = synsets[0]
    sentiwordnet_synset = sentiwordnet.senti_synset(synset.name())
    return [synset.name(), sentiwordnet_synset.pos_score(), sentiwordnet_synset.neg_score(), sentiwordnet_synset.obj_score()]

    pos = neg = obj = count = 0    
    
sentiment_score = []

for pos_value in df["pos_tags"]:
    sentiment_value = [get_sentiment(x,y) for (x,y) in pos_value]
    for score in sentiment_value:
        try:
            pos = pos + score[1]
            neg = neg + score[2]
        except:
            continue
    sentiment_score.append(pos - neg)
    pos = neg = 0    
    
df["sentiment_score"] = sentiment_score

overall = []
for i in range(len(df)):
    if df["sentiment_score"][i] >= 0.05:
        overall.append("Positive")
    elif df["sentiment_score"][i] <= -0.05:
        overall.append("Negative")
    else:
        overall.append("Neutral")
df["overall_sentiment"] = overall
print(df.head(10))

# Visualise the sentiment score results
# sns.set_theme(style = "whitegrid")
# sns.countplot(x=df["overall_sentiment"])
# plt.show() 

df["speech_text_new"] = df["post_lemmatization"].copy()

## Modelling

# from sklearn.feature_extraction.text import CountVectorizer

# vec = CountVectorizer()
# X = vec.fit_transform(data['reviews_text_new'])
# df = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
# df.head(3)

# vect = CountVectorizer()
# vect.fit(data['reviews_text_new'])
# vect.get_feature_names()
# # transform training data into a 'document-term matrix'
# simple_train_dtm = vect.transform(data['reviews_text_new'])
# print(simple_train_dtm)

# ### Creating a python object of the class CountVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# bow_counts = CountVectorizer(tokenizer= word_tokenize, # type of tokenization
#                              ngram_range=(1,3)) # number of n-grams

# bow_data = bow_counts.fit_transform(data['reviews_text_new'])

# from sklearn.model_selection import train_test_split
# X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data, # Features
#                                                                     data['Overall Sentiment'], # Target variable
#                                                                     test_size = 0.2, # 20% test size
#                                                                     random_state = 0) # random

# from sklearn.linear_model import LogisticRegression
# ### Training the model 
# lr_model_all = LogisticRegression() # Logistic regression
# lr_model_all.fit(X_train_bow, y_train_bow) # Fitting a logistic regression model

# ## Predicting the output
# test_pred_lr_all = lr_model_all.predict(X_test_bow) # Class prediction


# ## Calculate key performance metrics

# from sklearn.metrics import classification_report
# # Print a classification report
# print(classification_report(y_test_bow,test_pred_lr_all))

# from sklearn.feature_extraction.text import TfidfVectorizer
# noise_words = []
# ### Creating a python object of the class CountVectorizer
# tfidf_counts = TfidfVectorizer(tokenizer= word_tokenize, # type of tokenization
#                                stop_words=noise_words, # List of stopwords
#                                ngram_range=(1,1)) # number of n-grams

# tfidf_data = tfidf_counts.fit_transform(data['reviews_text_new'])

# X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data,
#                                                                             data['Overall Sentiment'],
#                                                                             test_size = 0.2,
#                                                                             random_state = 0)

# ### Setting up the model class
# lr_model_tf_idf = LogisticRegression()

# ## Training the model 
# lr_model_tf_idf.fit(X_train_tfidf,y_train_tfidf)

# ## Prediciting the results
# test_pred_lr_all = lr_model_tf_idf.predict(X_test_tfidf)

# ## Calculate key performance metrics

# # Print a classification report
# print(classification_report(y_test_tfidf,test_pred_lr_all))