import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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

def word_frequency_generator(comments_list, classification_list, spam_classification):
    word_freq = {}
    for i in range(len(comments_list)):
        if classification_list.iloc[i] == spam_classification:
            comment = preprocess_comments(comments_list.iloc[i])
            for word in comment:
                word_freq[word] = word_freq.get(word, 0) + 1

    return word_freq

def wordcloud_generator(word_freq_dict):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 1)
wordcloud_generator(spam_train_words_freq)

not_spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 0)
wordcloud_generator(not_spam_train_words_freq)

#creates sparse matrix of all the words in the comments
bag_of_words_transformer = CountVectorizer(analyzer=preprocess_comments).fit(comments_train_split)
comments_train_bow = bag_of_words_transformer.transform(comments_train_split)

#converts comments into tfidf word frequency
tfidf_transformer = TfidfTransformer().fit(comments_train_bow)
comments_train_tfidf = tfidf_transformer.transform(comments_train_bow)

#Multinomial Naive Bayes detection model
mnb_detection = MultinomialNB().fit(comments_train_tfidf, spam_classification_train)

#creates pipeline to process training data
mnb_pipeline = Pipeline([
    ('bag_of_words', CountVectorizer(analyzer=preprocess_comments)),
    ('tfidf', TfidfTransformer()),
    ('classification_model', MultinomialNB())
])

mnb_pipeline.fit(comments_train_split, spam_classification_train)
predictions_test = mnb_pipeline.predict(comments_test_split)
print(metrics.classification_report(spam_classification_test, predictions_test))