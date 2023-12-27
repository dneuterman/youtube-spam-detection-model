import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import nltk
from nltk.corpus import stopwords
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import metrics

#imports data from csv dataset
comments1 = pd.read_csv("./datasets/Youtube01-Psy.csv", encoding = "utf-8")
comments2 = pd.read_csv("./datasets/Youtube02-KatyPerry.csv", encoding = "utf-8")
comments3 = pd.read_csv("./datasets/Youtube03-LMFAO.csv", encoding = "utf-8")
comments4 = pd.read_csv("./datasets/Youtube04-Eminem.csv", encoding = "utf-8")
comments5 = pd.read_csv("./datasets/Youtube05-Shakira.csv", encoding = "utf-8")

#combines csv datasets and removes unnecessary columns
comments_dataframe = pd.concat([comments1, comments2, comments3, comments4, comments5], ignore_index=True)
comments_dataframe = comments_dataframe.drop(labels = ["AUTHOR", "DATE"], axis = 1)
comments_dataframe.columns = ["COMMENT_ID", "COMMENT", "SPAM_CLASSIFICATION"]

#adds length column to dataframe
comments_dataframe["LENGTH"] = comments_dataframe["COMMENT"].apply(len)

#creates pie chart showing the amounnt of spam comments in the dataset
spam_classification_series = comments_dataframe.groupby("SPAM_CLASSIFICATION")["SPAM_CLASSIFICATION"].count()
fig, ax = plt.subplots()
ax.pie(spam_classification_series, labels=["Not Spam", "Spam"], autopct='%1.1f%%')
plt.show()

#splits dataset into training and testing set 70/30
comments_train_split, comments_test_split, spam_classification_train, spam_classification_test = train_test_split(comments_dataframe["COMMENT"], comments_dataframe["SPAM_CLASSIFICATION"], test_size=0.3, random_state=42)

#function to remove stop words, punctuation, numbers and URLs from comment
def preprocess_comments(comment):
    #regular expression attributed to https://urlregex.com/index.html
    http_urlhyperlink_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    comment = re.sub(http_urlhyperlink_regex, 'urlsubstitute', comment)

    #regular expression attributed to https://uibakery.io/regex-library/html-regex-python
    html_tags_regex = "<(?:\"[^\"]*\"['\"]*|'[^']*'['\"]*|[^'\">])+>"
    comment = re.sub(html_tags_regex, '', comment)

    #removes \ufeff from comment
    comment = comment.replace("\ufeff", '')

    #removes punctuation
    clean_comment_array = []
    for char in comment:
        if char not in string.punctuation:
            clean_comment_array.append(char)

    clean_comment_array = ''.join(clean_comment_array).lower()
    clean_comment_array = clean_comment_array.split()

    #removes stop words. Isalpha removes numbers which also removes emjoi characters.
    word_list = []
    for word in clean_comment_array:
        if word.lower() not in stopwords.words('english') and word.isalpha():
            word_list.append(word)

    return word_list

#creates sparse matrix of all the words in the comments
bag_of_words_transformer = CountVectorizer(analyzer=preprocess_comments).fit(comments_dataframe["COMMENT"])
comments_bow = bag_of_words_transformer.transform(comments_dataframe["COMMENT"])

#converts comments into tfidf word frequency
tfidf_transformer = TfidfTransformer().fit(comments_bow)
comments_tfidf = tfidf_transformer.transform(comments_bow)

#Multinomial Naive Bayes detection model
mnb_detection = MultinomialNB().fit(comments_tfidf, comments_dataframe["SPAM_CLASSIFICATION"])

#creates pipeline to process training data
mnb_pipeline = Pipeline([
    ('bag_of_words', CountVectorizer(analyzer=preprocess_comments)),
    ('tfidf', TfidfTransformer()),
    ('classification_model', MultinomialNB())
])

mnb_pipeline.fit(comments_train_split, spam_classification_train)
predictions_test = mnb_pipeline.predict(comments_test_split)
print(metrics.classification_report(spam_classification_test, predictions_test))