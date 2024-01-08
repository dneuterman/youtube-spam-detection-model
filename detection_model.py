import numpy as np
import pandas as pd
import re
import json
import random

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
plt.clf()

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
    plt.clf()

def word_freq_bar_chart_generator(word_freq_dict, chart_size, chart_filename):
    words_to_plot = []
    words_to_plot_freq = {}
    for key, value in word_freq_dict.items():
        words_to_plot.append(key)
        words_to_plot_freq[value] = len(words_to_plot) - 1

    sorted_dict = sorted(words_to_plot_freq, reverse=True)
    plot_x, plot_y = [], []

    for i in range(chart_size):
        plot_y.append(words_to_plot[words_to_plot_freq[sorted_dict[i]]])
        plot_x.append(sorted_dict[i])

    sns.set(style="darkgrid")
    sns.barplot(x=plot_x, y=plot_y, color="b", orient="h")
    plt.xlabel("Word Frequency")
    plt.savefig(f"./static/plots/{chart_filename}.svg")
    plt.clf()

#creates small sample from test dataset to use for later
def create_sample_test_dataset(test_dataset, size):
    comments_set = set()
    comments_array = []
    while size > 0:
        rand_comment = random.randrange(0, len(test_dataset))
        if rand_comment not in comments_set:
            comments_array.append(test_dataset[rand_comment])
            size -= 1
    return comments_array

comments_train_split_json = convert_initial_to_json(comments_train_split)
comments_test_split_json = convert_initial_to_json(comments_test_split)
sample_comments_json = create_sample_test_dataset(comments_test_split_json, 10)

save_dataset_to_json(comments_train_split_json, "training-comments-dataset.json")
save_dataset_to_json(comments_test_split_json, "test-comments-dataset.json")
save_dataset_to_json(sample_comments_json, "sample-comments.json")

spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 1)
not_spam_train_words_freq = word_frequency_generator(comments_train_split, spam_classification_train, 0)
word_freq_bar_chart_generator(spam_train_words_freq, 10, "spam-words-training-bar-chart")
word_freq_bar_chart_generator(not_spam_train_words_freq, 10, "not-spam-words-training-bar-chart")

wordcloud_generator(spam_train_words_freq, "spam-words-training-wordcloud")
wordcloud_generator(not_spam_train_words_freq, "not-spam-words-training-wordcloud")

#creates pipeline to process training data
mnb_pipeline = PipelineProcessor()

mnb_pipeline.fit_model(comments_train_split, spam_classification_train)
mnb_pipeline.predict_model(comments_test_split)
print(metrics.classification_report(spam_classification_test, mnb_pipeline.prediction))

def confusion_matrix_generator(classification_test, predicted_test):
    confusion_matrix = metrics.confusion_matrix(classification_test, predicted_test)
    print(confusion_matrix)

    labels = ["True Neg", "False Pos", "False Neg", "True Pos"]
    flattened_cf = confusion_matrix.flatten()
    sum_of_cf= np.sum(confusion_matrix)
    for i in range(len(labels)):
        labels[i] = f"{labels[i]}\n" + "{0:.2%}".format(flattened_cf[i]/sum_of_cf)

    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')
    plt.ylabel("Actual Values")
    plt.xlabel("Predicted Values")
    plt.savefig(f"./static/plots/test-prediction-cf-matrix.svg")
    plt.clf()

confusion_matrix_generator(spam_classification_test, mnb_pipeline.prediction)
accuracy = metrics.accuracy_score(spam_classification_test, mnb_pipeline.prediction)
macro_precision_average = metrics.precision_score(spam_classification_test, mnb_pipeline.prediction, average="macro")
macro_recall_average = metrics.recall_score(spam_classification_test, mnb_pipeline.prediction, average="macro")
macro_f1_average = metrics.f1_score(spam_classification_test, mnb_pipeline.prediction, average="macro")

print(metrics.classification_report(spam_classification_test, mnb_pipeline.prediction))
classifcation_report_dict = metrics.classification_report(spam_classification_test, mnb_pipeline.prediction, output_dict=True)
print(classifcation_report_dict)
print(f"Accuracy Score: {classifcation_report_dict['accuracy']}")
print(f"Precision Macro Average: {classifcation_report_dict['macro avg']['precision']}")
print(f"Recall Macro Average: {classifcation_report_dict['macro avg']['recall']}")
print(f"F1-Score Macro Average: {classifcation_report_dict['macro avg']['f1-score']}")