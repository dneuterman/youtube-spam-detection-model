from flask import Flask, render_template
import json
import random

with open("./datasets/json/training-comments-dataset.json", "r") as file:
    training_comments_json = json.load(file)
with open("./datasets/json/test-comments-dataset.json", "r") as file:
    test_comments_json = json.load(file)
with open("./datasets/json/sample-comments.json", "r") as file:
    sample_comments_json = json.load(file)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/dataset/")
def dataset():
    sample_comment = ""
    clean_sample_comment = ""
    sample_comment_classification = ""
    while(clean_sample_comment == ""):
        rand_comment = random.randrange(0, len(training_comments_json))
        sample_comment = training_comments_json[rand_comment]["comment"]
        clean_sample_comment = training_comments_json[rand_comment]["clean_comment"]
        sample_comment_classification = str(training_comments_json[rand_comment]["actual_classification"])
    return render_template('dataset.html', sample_comment = sample_comment, clean_sample_comment = clean_sample_comment, sample_comment_classification = sample_comment_classification)

@app.route("/training/")
def training():
    return render_template('training.html')

@app.route("/prediction/")
def prediction():
    return render_template('prediction.html', sample_comments = sample_comments_json)