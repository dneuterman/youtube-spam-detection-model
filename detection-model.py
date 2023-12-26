import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import nltk
from nltk.corpus import stopwords
import string

comments1 = pd.read_csv("./datasets/Youtube01-Psy.csv", encoding = "latin-1")
comments2 = pd.read_csv("./datasets/Youtube02-KatyPerry.csv", encoding = "latin-1")
comments3 = pd.read_csv("./datasets/Youtube03-LMFAO.csv", encoding = "latin-1")
comments4 = pd.read_csv("./datasets/Youtube04-Eminem.csv", encoding = "latin-1")
comments5 = pd.read_csv("./datasets/Youtube05-Shakira.csv", encoding = "latin-1")

comments_dataframe = pd.concat([comments1, comments2, comments3, comments4, comments5])
comments_dataframe = comments_dataframe.drop(labels = ["AUTHOR", "DATE"], axis = 1)
comments_dataframe.columns = ["COMMENT_ID", "COMMENT", "SPAM_CLASSIFICATION"]

comments_dataframe["LENGTH"] = comments_dataframe["COMMENT"].apply(len)

def preprocess_comments(comment):
    clean_comment_array = []
    for char in comment:
        if char not in string.punctuation:
            clean_comment_array.append(char)

    clean_comment_array = ''.join(clean_comment_array).lower()
    clean_comment_array = clean_comment_array.split()

    word_list = []
    for word in clean_comment_array:
        if word.lower() not in stopwords.words('english') and word.isalpha():
            word_list.append(word)

    return word_list

message_list = comments_dataframe["COMMENT"]
message_words = preprocess_comments(message_list)
print(message_words)
