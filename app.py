from flask import Flask, render_template, request
import json
import random
from pipeline_processor import PipelineProcessor

with open("./datasets/json/training-comments-dataset.json", "r") as file:
    training_comments_json = json.load(file)
with open("./datasets/json/test-comments-dataset.json", "r") as file:
    test_comments_json = json.load(file)
with open("./datasets/json/sample-comments.json", "r") as file:
    sample_comments_json = json.load(file)

mnb_pipeline = PipelineProcessor()

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

@app.route("/prediction/", methods = ["GET", "POST"])
def prediction():
    if request.method == "POST":
        result = []
        comments_to_predict = []
        if "user-comment-btn" in request.form:
            result.append({
                "comment": request.form.get('comment')
            })
        if "sample-comments-btn" in request.form:
            result = sample_comments_json

        for i in range(len(result)):
            comments_to_predict.append(result[i]["comment"])

        mnb_pipeline.predict_model(comments_to_predict)
        for i in range(len(result)):
            result[i]["predicted_classification"] = mnb_pipeline.prediction[i]

        return render_template('completed-prediction.html', result = result)
    else:
        return render_template('prediction.html', sample_comments = sample_comments_json)