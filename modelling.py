import pandas as pd
import nltk
from nltk.tokenize import word_tokenize 
from joblib import dump, load



_JSON_INPUT_FOLDER = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\json"
_JSON_TEST_FILE = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\json\\2013-01-08.json"
_PREPROCESSED_DATA_INPUT = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\preprocessed_data.csv"

### Modelling

data = pd.read_csv(_PREPROCESSED_DATA_INPUT, usecols={"id":str, "speakername":str, "url":str, "vote":str, "date":str, "speech_text_new":str, "sentiment_score":float, "overall_sentiment":str})
print(data)

# Transform training data into a 'document-term matrix'
# from sklearn.feature_extraction.text import CountVectorizer
# print("Transforming training data into a document-term matrix...")
# vect_doc_term = CountVectorizer()
# vect_doc_term.fit(data["speech_text_new"].values.astype(str))
# vect_doc_term.get_feature_names_out()
# simple_train_dtm = vect_doc_term.transform(data["speech_text_new"].values.astype(str))
# print(simple_train_dtm)

# Create CountVectorizer object for Bag of Words data
# bow_counts = CountVectorizer(tokenizer = word_tokenize, ngram_range=(1, 3))
# bow_data = bow_counts.fit_transform(data["speech_text_new"])
# Divide data into training and test sets
# from sklearn.model_selection import train_test_split
# X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(bow_data, data["overall_sentiment"], test_size = 0.2, random_state = 0)

# Apply TF-IDF model to data
print("Applying TF-IDF model...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
noise_words = []
tfidf_counts = TfidfVectorizer(tokenizer = word_tokenize, stop_words = noise_words, ngram_range=(1,2))
tfidf_data = tfidf_counts.fit_transform(data["speech_text_new"].values.astype(str))
# Divide data into training and test sets
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data, data["overall_sentiment"], test_size = 0.8, random_state = 0)

# Train baseline MultinomialNB model
print("Training MultinomialNB model...")
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
nb_model_tf_idf = MultinomialNB()
nb_model_tf_idf.fit(X_train_tfidf, y_train_tfidf)
# Predict with MultinomialNB model
test_pred_nb_all = nb_model_tf_idf.predict(X_test_tfidf)
# MultinomialNB performance metrics
print("\nMultinomialNB Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_tfidf, test_pred_nb_all))
print("\nClassification Report:")
print(classification_report(y_test_tfidf, test_pred_nb_all))
print("\nAccuracy Score:")
print(accuracy_score(y_test_tfidf, test_pred_nb_all))

# Train LogisticRegression model
print("Training LogisticRegression model...")
from sklearn.linear_model import LogisticRegression
lr_model_tf_idf = LogisticRegression(verbose = 1, max_iter = 1000)
lr_model_tf_idf.fit(X_train_tfidf, y_train_tfidf)
# Predict with LogisticRegression model
test_pred_lr_all = lr_model_tf_idf.predict(X_test_tfidf)
# LogisticRegression performance metrics
print("\nLogisticRegression Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_tfidf, test_pred_lr_all))
print("\nClassification Report:")
print(classification_report(y_test_tfidf, test_pred_lr_all))
print("\nAccuracy Score:")
print(accuracy_score(y_test_tfidf, test_pred_lr_all))

# Train SGDClassifier model
print("\nTraining SGDClassifier model...")
from sklearn.linear_model import SGDClassifier
sgd_model_tf_idf = SGDClassifier(verbose = 1, penalty = "elasticnet", loss = "log_loss")
sgd_model_tf_idf.fit(X_train_tfidf, y_train_tfidf)
# Predict with SGDClassifier model
test_pred_sgd_all = sgd_model_tf_idf.predict(X_test_tfidf)
# SGDClassifier performance metrics
print("\nSGDClassifier Results:")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_tfidf, test_pred_sgd_all))
print("\nClassification Report:")
print(classification_report(y_test_tfidf, test_pred_sgd_all))
print("\nAccuracy Score:")
print(accuracy_score(y_test_tfidf, test_pred_sgd_all))

# Pickle the models
print("\nPickling MultinomialNB model...")
dump(nb_model_tf_idf, ".\\models\\nb_model.joblib")
print("Pickling LogisticRegression model...")
dump(lr_model_tf_idf, ".\\models\\lr_model.joblib")
print("Pickling SGDClassifier model...")
dump(sgd_model_tf_idf, ".\\models\\sgd_model.joblib")