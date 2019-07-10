"""
jsonでPOST requestを受けとり、予測した結果をjsonにして返す
input:  {1: [11種の特徴量], 2: [11種の特徴量], ...}
output: {1: 0 or 1, 2:0 or 1, ...}
"""
import flask 
import json
from flask import request, Flask, jsonify
import pandas as pd
import numpy as np
import pickle
from predict import preprocess_test
from util import *

app = Flask(__name__)

COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", 
           "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
MODEL_PATH = "forest.pkl"
TRAIN = pd.read_csv("data/train.csv")

def transform_json2df(data: dict) -> pd.DataFrame:
    test_df = pd.DataFrame.from_dict(data)
    return test_df

def load_model(path):
    with open(path, mode="rb") as f:
        model = pickle.load(f)
    return model

def transform_df2dict(pred_Y: np.ndarray, passenger_id: np.ndarray):
    pred_Y_dict = {}
    for pid, y in zip(passenger_id, pred_Y):
        pred_Y_dict[pid] = y
    return pred_Y_dict

def to_string(array: np.ndarray):
    list_ = array.tolist()
    size = len(list_)
    for i in range(size):
        list_[i] = str(list_[i])
    return list_

@app.route("/predict", methods=["POST"])
def predict():
    assert(request.method == "POST")
    assert(request.headers["Content-Type"] == "application/json")
    test_data = request.json
    test_df = transform_json2df(test_data)
    passenger_id = test_df.PassengerId.values
    assert(test_df.shape[0] == len(test_data["PassengerId"]))

    test_df = preprocess_test(test_df, TRAIN)
    model = load_model(MODEL_PATH)
    pred_Y = model.predict(test_df)

    pred_Y = to_string(pred_Y)
    passenger_id = to_string(passenger_id)
    pred_Y_dict = transform_df2dict(pred_Y, passenger_id)

    return json.dumps(stringify_keys(pred_Y_dict))
    # return "OK"


if __name__ == '__main__':
    app.run()