import numpy as np
import pandas as pd

comments1 = pd.read_csv("./datasets/Youtube01-Psy.csv", encoding = "latin-1")
comments2 = pd.read_csv("./datasets/Youtube02-KatyPerry.csv", encoding = "latin-1")
comments3 = pd.read_csv("./datasets/Youtube03-LMFAO.csv", encoding = "latin-1")
comments4 = pd.read_csv("./datasets/Youtube04-Eminem.csv", encoding = "latin-1")
comments5 = pd.read_csv("./datasets/Youtube05-Shakira.csv", encoding = "latin-1")

comments_dataframe = pd.concat([comments1, comments2, comments3, comments4, comments5])
comments_dataframe = comments_dataframe.drop(labels = ["COMMENT_ID", "AUTHOR", "DATE"], axis = 1)
comments_dataframe.columns = ["COMMENT", "SPAM CLASSIFICATION"]