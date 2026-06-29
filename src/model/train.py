import argparse
import glob
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


def split_data(df):
    feature_cols = [
        "Pregnancies", "PlasmaGlucose", "DiastolicBloodPressure",
        "TricepsThickness", "SerumInsulin", "BMI", "DiabetesPedigree", "Age",
    ]
    X = df[feature_cols].values
    y = df["Diabetic"].values
    return train_test_split(X, y, test_size=0.30, random_state=0)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    mlflow.sklearn.autolog()
    model = LogisticRegression(C=1 / reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Test accuracy: {accuracy:.4f}")
    return model


def main(args):
    mlflow.set_experiment("diabetes-classification")
    with mlflow.start_run() as run:
        mlflow.log_param("reg_rate", args.reg_rate)
        df = get_csvs_df(args.training_data)
        X_train, X_test, y_train, y_test = split_data(df)
        train_model(args.reg_rate, X_train, X_test, y_train, y_test)
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.register_model(model_uri, "diabetes-classifier")
        print(f"Run ID: {run.info.run_id}")
        print(f"Model registered as: diabetes-classifier")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", type=str, default="data")
    parser.add_argument("--reg_rate", type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
