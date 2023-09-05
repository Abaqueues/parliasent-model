import pandas as pd
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize 
from sklearn.model_selection import cross_val_score


_PREPROCESSED_DATA_OUTPUT = "E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\preprocessed_data.csv"

data = pd.read_csv(_PREPROCESSED_DATA_OUTPUT, usecols={"id":str, "speakername":str, "url":str, "vote":str, "date":str, "speech_text_new":str, "sentiment_score":float, "overall_sentiment":str})
print(data)

# Transform data into a 'document-term matrix'
# from sklearn.feature_extraction.text import CountVectorizer
# print("Transforming training data into a document-term matrix...")
# vect_doc_term = CountVectorizer()
# vect_doc_term.fit(data["speech_text_new"].values.astype(str))
# vect_doc_term.get_feature_names_out()
# simple_train_dtm = vect_doc_term.transform(data["speech_text_new"].values.astype(str))

# Apply TF-IDF model to data
print("Applying TF-IDF model...")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
noise_words = []
tfidf_counts = TfidfVectorizer(tokenizer = word_tokenize, stop_words = noise_words, ngram_range=(1,2))
tfidf_data = tfidf_counts.fit_transform(data["speech_text_new"].values.astype(str))
# Divide data into training and test sets
# X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(tfidf_data, data["overall_sentiment"], test_size = 1.0, random_state = 0)

# Load pickled model
nb_model_tf_idf = load(".\\models\\nb_model.joblib")
lr_model_tf_idf = load(".\\models\\lr_model.joblib")
sgd_model_tf_idf = load(".\\models\\sgd_model.joblib")

# Predict with MultinomialNB model
print("\nPredicting with MultinomialNB model...")
pred_nb_all = nb_model_tf_idf.predict(tfidf_data)
data["nb_prediction"] = pred_nb_all

scores = cross_val_score(nb_model_tf_idf, tfidf_data, data["overall_sentiment"], cv=5)
print("\nMultinomialNB Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Predict with LogisticRegression model
print("\nPredicting with LogisticRegression model...")
pred_lr_all = lr_model_tf_idf.predict(tfidf_data)
data["lr_prediction"] = pred_lr_all

scores = cross_val_score(lr_model_tf_idf, tfidf_data, data["overall_sentiment"], cv=5)
print("\nLinearRegression Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# Predict with SGDClassifier model
print("\nPredicting with SGDClassifier model...")
pred_sgd_all = sgd_model_tf_idf.predict(tfidf_data)
data["sgd_prediction"] = pred_sgd_all

scores = cross_val_score(sgd_model_tf_idf, tfidf_data, data["overall_sentiment"], cv=5)
print("\nSGDClassifier Cross Validation Results:")
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

print(data)

data.to_csv("E:\\Documents\\Education\\PG University of Birmingham\\MSc Computer Science\\Summer Semester\\MSc Projects\\Project Files\\Dataset\\final\\predicted_data.csv", encoding = "utf-8")