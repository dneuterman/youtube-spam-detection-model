from flask import Flask, render_template, request
import json
import random
import os
from pipeline_processor import PipelineProcessor

with open("./datasets/json/training-comments-dataset.json", "r") as file:
    training_comments_json = json.load(file)
with open("./datasets/json/test-comments-dataset.json", "r") as file:
    test_comments_json = json.load(file)
with open("./datasets/json/sample-comments.json", "r") as file:
    sample_comments_json = json.load(file)

class PredictedComments:
    def __init__(self):
        self.comments = None

    def get_comments(self):
        return self.comments
    
    def set_comments(self, comments):
        self.comments = comments

mnb_pipeline = PipelineProcessor()
predicted_comments = PredictedComments()

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
        if "download-spam-comments" in request.form:
            comments_to_delete = []
            comments = predicted_comments.get_comments()
            for i in range(len(comments)):
                if comments[i]["predicted_classification"] == 1:
                    del comments[i]["clean_comment"]
                    comments_to_delete.append(comments[i])

            filename = "comments_to_delete.json"

            with open(f"./datasets/saved/{filename}", "w") as file:
                json.dump(comments_to_delete, file)

            filepath = f"{os.path.dirname(app.instance_path)}\datasets\saved\{filename}"

            return render_template('download-comments.html', filepath = filepath)
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
            result[i]["predicted_classification"] = int(mnb_pipeline.prediction[i])
        predicted_comments.set_comments(result)
        return render_template('completed-prediction.html', result = result)
    else:
        return render_template('prediction.html', sample_comments = sample_comments_json)