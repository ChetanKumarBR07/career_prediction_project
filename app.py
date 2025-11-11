# app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("career_pipeline.pkl")

career_labels = [
    "Software Engineer", "Data Scientist", "Chartered Accountant", "Journalist",
    "AI Researcher", "Financial Analyst", "Author", "Engineer", "Artist", "Investment Banker"
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/inputs')
def input_page():
    return render_template('inputs.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form.get("age"))
    stream = request.form.get("stream")
    math = int(request.form.get("math"))
    physics = int(request.form.get("physics"))
    chemistry = int(request.form.get("chemistry"))
    skills = request.form.get("skills")
    interests = request.form.get("interests")
    input_data = pd.DataFrame([[age, stream, math, physics, chemistry, skills, interests]],
                              columns=["Age", "Stream", "Math", "Physics", "Chemistry", "Skills", "Interests"])
    predicted_career_index = model.predict(input_data)[0]
    predicted_career = career_labels[predicted_career_index]

    return render_template('result.html', result=predicted_career)

if __name__ == '__main__':
    app.run(debug=True)
