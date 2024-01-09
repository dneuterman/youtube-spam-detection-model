from flask import Flask, render_template, request, redirect
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
with open("./datasets/json/test-comments-classification-report.json", "r") as file:
    test_classification_report = json.load(file)

class PredictedComments:
    def __init__(self):
        self.comments = None

    def get_comments(self):
        return self.comments
    
    def set_comments(self, comments):
        self.comments = comments

mnb_pipeline = PipelineProcessor()
predicted_comments = PredictedComments()

UPLOAD_FOLDER = 'datasets/json'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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
    return render_template('training.html', classification_report = test_classification_report)

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

        additional_param_flag = None
        if "upload-comments-btn" in request.form:
            uploaded_comments = request.files['upload-comment-file']
            if uploaded_comments.filename == '':
                return redirect(request.url)
            
            filename = "./datasets/json/uploaded-comments.json"
            uploaded_comments.save("./datasets/json/uploaded-comments.json")
            with open("./datasets/json/uploaded-comments.json", "r") as file:
                result = json.load(file)
            additional_param_flag = "uploaded-comments"
        if "user-comment-btn" in request.form:
            result.append({
                "comment": request.form.get('comment')
            })
            additional_param_flag = "user-comments"
        if "sample-comments-btn" in request.form:
            result = sample_comments_json
            additional_param_flag = "sample-comments"

        for i in range(len(result)):
            comments_to_predict.append(result[i]["comment"])

        mnb_pipeline.predict_model(comments_to_predict)
        for i in range(len(result)):
            result[i]["predicted_classification"] = int(mnb_pipeline.prediction[i])
        predicted_comments.set_comments(result)
        return render_template('completed-prediction.html', result = json.dumps(result), form_button_name = additional_param_flag)
    else:
        return render_template('prediction.html', sample_comments = sample_comments_json)