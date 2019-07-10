"""
jsonでPOST requestを受けとり、予測した結果をjsonにして返す
input:  {1: [11種の特徴量], 2: [11種の特徴量], ...}
output: {1: 0 or 1, 2:0 or 1, ...}
"""
import flask 
import json
from flask import request
import pandas as pd
import numpy as np
import pickle

app = flask.Flask(__name__)

COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", 
           "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
MODEL_PATH = "forest.pkl"

def transform_json2df(data: dict) -> pd.DataFrame:
    test_df = pd.DataFrame(columns=COLUMNS)
    for key_, value_ in data.items():
        test_df.loc(key_) = value_
    return test_df

def load_model(path):
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model

@app.route("/predict", method=["POST"])
def predict():
    assert(request.method == "POST")
    test_data = request.data.decode('utf-8')
    test_data = json.loads(test_data)
    test_df = transform_json2df(test_data)
    assert(test_df.shape[0] == len(test_data))

    model = load_model(MODEL_PATH)


