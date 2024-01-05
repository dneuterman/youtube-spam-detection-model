import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle

from comments_preprocessor import comments_preprocessor

#Takes pandas series as input, fits model and dumps to pickle file
def fit_model(training_split, training_classification):
    mnb_pipeline = Pipeline([
        ('bag_of_words', CountVectorizer(analyzer=comments_preprocessor)),
        ('tfidf', TfidfTransformer()),
        ('classification_model', MultinomialNB())
    ])

    mnb_pipeline.fit(training_split, training_classification)
    filename = "mnb_model.pickle"
    with open(filename, 'wb') as file:
        pickle.dump(mnb_pipeline, file)

def predict_model(test_split):
    filename = "mnb_model.pickle"
    with open(filename, 'rb') as file:
        mnb_model = pickle.load(file)
    return mnb_model.predict(test_split)

