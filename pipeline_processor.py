import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pickle

from comments_preprocessor import comments_preprocessor

class PipelineProcessor:
    def __init__(self):
        self.model = None
        self.filename = "mnb_model.pickle"

    #Takes pandas series as input, fits model and dumps to pickle file
    def fit_model(self, training_split, training_classification):
        mnb_pipeline = Pipeline([
            ('bag_of_words', CountVectorizer(analyzer=comments_preprocessor)),
            ('tfidf', TfidfTransformer()),
            ('classification_model', MultinomialNB())
        ])

        mnb_pipeline.fit(training_split, training_classification)
        with open(self.filename, 'wb') as file:
            pickle.dump(mnb_pipeline, file)

    def predict_model(self, test_split):
        if self.model == None:
            with open(self.filename, 'rb') as file:
                self.model = pickle.load(file)
        return self.model.predict(test_split)

