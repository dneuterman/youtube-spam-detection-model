import numpy as np
import pandas as pd
import re
import json

import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud, ImageColorGenerator

from sklearn.model_selection import train_test_split
from sklearn import metrics

from pipeline_processor import PipelineProcessor
from comments_preprocessor import comments_preprocessor

#imports data from csv dataset
comments1 = pd.read_csv("./datasets/raw/Youtube01-Psy.csv", encoding = "utf-8")
comments2 = pd.read_csv("./datasets/raw/Youtube02-KatyPerry.csv", encoding = "utf-8")
comments3 = pd.read_csv("./datasets/raw/Youtube03-LMFAO.csv", encoding = "utf-8")
comments4 = pd.read_csv("./datasets/raw/Youtube04-Eminem.csv", encoding = "utf-8")
comments5 = pd.read_csv("./datasets/raw/Youtube05-Shakira.csv", encoding = "utf-8")

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
plt.savefig("./static/plots/comments-classification-pie-chart.svg")

#splits dataset into training and testing set 70/30
comments_train_split, comments_test_split, spam_classification_train, spam_classification_test = train_test_split(comments_dataframe["COMMENT"], comments_dataframe["SPAM_CLASSIFICATION"], test_size=0.3, random_state=31)

def convert_initial_to_json(split_comments_series):
    comments_index_list = split_comments_series.index.tolist()
    comments_array = []
    for index in comments_index_list:
        current_comment = comments_dataframe["COMMENT"][index]
        comment_info = {
            "comment_id": comments_dataframe["COMMENT_ID"][index],
            "comment": current_comment,
            "clean_comment": " ".join(comments_preprocessor(current_comment)),
            "actual_classification": int(comments_dataframe["SPAM_CLASSIFICATION"][index])
        }
        comments_array.append(comment_info)
    return comments_array

def save_dataset_to_json(data, filename):
    with open(f"./datasets/json/{filename}", "w") as file:
        json.dump(data, file)

def word_frequency_generator(comments_list, classification_list, spam_classification):
    word_freq = {}
    for i in range(len(comments_list)):
        if classification_list.iloc[i] == spam_classification:
            comment = comments_preprocessor(comments_list.iloc[i])
            for word in comment:
                word_freq[word] = word_freq.get(word, 0) + 1

    return word_freq

def wordcloud_generator(word_freq_dict, wordcloud_filename):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(f"./static/plots/{wordcloud_filename}.svg")

#creates small sample from test dataset to use for later
def create_sample_test_dataset(test_dataset):
    pass

comments_train_split_json = convert_initial_to_json(comments_train_split)
comments_test_split_json = convert_initial_to_json(comments_test_split)

save_dataset_to_json(convert_initial_to_json(comments_train_split), "training-comments-dataset.json")
save_dataset_to_json(convert_initial_to_json(comments_test_split), "test-comments-dataset.json")

spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 1)
wordcloud_generator(spam_train_words_freq, "spam-words-training-wordcloud")

not_spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 0)
wordcloud_generator(not_spam_train_words_freq, "not-spam-words-training-wordcloud")

#creates sparse matrix of all the words in the comments
# bag_of_words_transformer = CountVectorizer(analyzer=preprocess_comments).fit(comments_train_split)
# comments_train_bow = bag_of_words_transformer.transform(comments_train_split)

#converts comments into tfidf word frequency
# tfidf_transformer = TfidfTransformer().fit(comments_train_bow)
# comments_train_tfidf = tfidf_transformer.transform(comments_train_bow)

#Multinomial Naive Bayes detection model
# mnb_detection = MultinomialNB().fit(comments_train_tfidf, spam_classification_train)

#creates pipeline to process training data
mnb_pipeline = PipelineProcessor()

mnb_pipeline.fit_model(comments_train_split, spam_classification_train)
mnb_pipeline.predict_model(comments_test_split)
print(metrics.classification_report(spam_classification_test, mnb_pipeline.prediction))