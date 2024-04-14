from flask import Flask,render_template,request,session
import os
import numpy as np
from werkzeug.utils import secure_filename
from model.utils import main

UPLOAD_FOLDER = os.path.join('static','uploads')
ALLOWED_EXTENSIONS = {'jpg' ,'png','jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "paraproj"

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/show',methods=['POST','GET'])
def show():
    if request.method=="POST":
        name = request.form["FirstName"]
        age = request.form["age"]
        height = request.form["height"]
        weight = request.form["weight"]
        img = request.files["image"]
        img_filename = secure_filename(img.filename)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        session['uploaded_img_filepath'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        img_filepath = session.get('uploaded_img_filepath', None)
        prediction = main.prediction(img_filepath)
        remark = inference(prediction)
        return render_template('output.html',image=img_filepath, name=name, age=age, height=height,weight=weight,
                                predict = [round(i, 2) for i in prediction],remarks=remark)

def inference(cobb):
    rule1 = cobb[0] < 20 and cobb[1] < 20 and cobb[2] < 20
    rule2 = cobb[0] >= 20 and cobb[1] <= 40 and cobb[2] <= 40
    rule3 = cobb[0] >= 40 or cobb[1] >= 40 or cobb[2] >= 60

    if rule1 and not rule2 and not rule3:
        severity = ["You are not affected with scoliosis.", "Walking, Jogging"]
    elif not rule1 and rule2 and not rule3:
        severity = ["You are in the starting stage of scoliosis and need to follow exercises.",
                    "Stretching exercises, Swimming, Yoga"]
    elif not rule1 and not rule2 and rule3:
        severity = ["Your scoliosis severity is moderate. Consult with your Physician.",
                    "Core strengthening exercises, Resistance training, Pilates"]
    elif not rule1 and rule2 and rule3:
        severity = ["Your scoliosis severity is high. Consult with your Physician.",
                    "Physical therapy, Bracing, Surgery"]
    elif not rule1 and not rule2 and not rule3:
        severity = ["Your scoliosis severity is severe. Immediate consultation with a specialist is required.",
                    "Physical therapy, Bracing, Surgery"]
    else:
        severity = ["Please consult with your physician for a proper evaluation of your scoliosis.", "-"]

    return severity

if __name__ == '__main__':
    app.run(debug=True)