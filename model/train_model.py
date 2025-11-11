# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


data = {
    "Age": [17, 18, 17, 19, 18, 17, 18, 19, 17, 18],
    "Stream": ["Science", "Science", "Commerce", "Arts", "Science", "Commerce", "Arts", "Science", "Arts", "Commerce"],
    "Math": [85, 72, 60, 40, 90, 50, 30, 80, 20, 55],
    "Physics": [80, 65, 50, 30, 85, 40, 20, 78, 25, 45],
    "Chemistry": [75, 70, 55, 35, 88, 45, 25, 82, 30, 50],
    "Skills": ["Coding", "Coding", "Finance", "Writing", "Coding", "Finance", "Writing", "Engineering", "Arts", "Finance"],
    "Interests": ["AI", "Software", "Business", "Literature", "Data Science", "Economics", "History", "Physics", "Painting", "Banking"],
    "Career": ["Software Engineer", "Data Scientist", "Chartered Accountant", "Journalist", "AI Researcher", "Financial Analyst", 
               "Author", "Engineer", "Artist", "Investment Banker"]
}

df = pd.DataFrame(data)

preprocessor = ColumnTransformer(
    transformers=[
        ('stream', OneHotEncoder(handle_unknown='ignore'), ['Stream']),
        ('skills', OneHotEncoder(handle_unknown='ignore'), ['Skills']),
        ('interests', OneHotEncoder(handle_unknown='ignore'), ['Interests'])
    ],
    remainder='passthrough'
)

X = df.drop(columns=["Career"])
y = df["Career"]

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier())
])

model.fit(X, y)

joblib.dump(model, 'career_pipeline.pkl')

print("Model trained and saved as 'career_pipeline.pkl'.")
