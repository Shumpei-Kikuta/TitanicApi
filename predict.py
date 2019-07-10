import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

DROP_COLUMNS = ["PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin"]

def generate_onehot_encoding(data: pd.DataFrame, column_name: str, drop=True):
    onehot_repr = pd.get_dummies(data[column_name])
    data = data.join(onehot_repr)
    data.drop(column_name, axis=1, inplace=True)
    return data

def preprocess_train(train: pd.DataFrame) -> pd.DataFrame:
    # 関係ないカラムを落とす
    train.drop(DROP_COLUMNS, axis=1, inplace=True)

    # 欠損値埋め
    mean_age = train.Age.mean()
    train["Age"] = train.Age.fillna(mean_age)
    train["Embarked"] = train.Embarked.fillna("S")

    # one-hot encoding
    train = generate_onehot_encoding(train, "Pclass")
    train = train.rename(columns={1: "class1", 2: "class2", 3: "class3"})
    train = generate_onehot_encoding(train, "Sex")
    train = generate_onehot_encoding(train, "Embarked")

    return train

def preprocess_test(test: pd.DataFrame, train:pd.DataFrame) -> pd.DataFrame:
    # 関係ないカラムを落とす
    test.drop(DROP_COLUMNS, axis=1, inplace=True)
    
    # 欠損値埋め
    mean_age = train.Age.mean()
    test["Age"]  = test.Age.fillna(mean_age)
    mean_fare = train.Fare.mean()
    test["Fare"] = test.Fare.fillna(mean_fare)

    # one-hot encoding
    test = generate_onehot_encoding(test, "Pclass")
    test = test.rename(columns={1: "class1", 2: "class2", 3: "class3"})
    test = generate_onehot_encoding(test, "Sex")
    test = generate_onehot_encoding(test, "Embarked")

    return test



def main():
    # データ読み込み
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    #前処理
    train = preprocess_train(train)
    test = preprocess_test(test, train)

    # modeling 
    train_X = train.drop("Survived", axis=1)
    train_Y = train.Survived
    test_X = test

    forest = RandomForestClassifier(n_estimators=100)
    forest.fit(train_X, train_Y)

    test_Y = forest.predict(test_X)

    # submission
    test = pd.read_csv("data/test.csv")
    submission_df = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": test_Y})
    submission_df.to_csv("submission.csv", index=False)

    #save model
    with open("forest.pkl", mode="wb") as f:
        pickle.dump(forest, f)


if __name__ == '__main__':
    main()