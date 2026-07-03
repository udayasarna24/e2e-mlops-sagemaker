import argparse
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import joblib


def model_fn(model_dir):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model


def input_fn(request_body, request_content_type):
    import io
    import json

    if request_content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None)
        return df.values

    elif request_content_type == "application/json":
        data = json.loads(request_body)

        if "instances" in data:
            return pd.DataFrame(data["instances"]).values
        elif isinstance(data, dict):
            return pd.DataFrame([data]).values
        else:
            raise ValueError("JSON must contain 'instances' key or be a flat dict")

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, content_type):
    import io
    import json
    import numpy as np

    preds = np.atleast_1d(prediction).tolist()

    if content_type == "application/json":
        response_body = json.dumps({"predictions": preds})
        return response_body, "application/json"   # ✅ explicit content-type

    else:
        output = io.StringIO()
        for p in preds:
            output.write(f"{p}\n")
        return output.getvalue(), "text/csv"         # ✅ explicit content-type


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    # Hyperparameters — passed by SageMaker from the estimator's
    # hyperparameters={} dict, so they show up in Studio automatically
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    print(f"✅ Hyperparameters : n_estimators={args.n_estimators}, "
          f"random_state={args.random_state}")

    train_file = os.path.join(args.train, "train.csv")
    print(f"✅ Files : {os.listdir(args.train)}")

    df = pd.read_csv(train_file)
    print(f"✅ Shape : {df.shape}")

    X = df.drop("income_above_50k", axis=1)
    y = df["income_above_50k"]

    # Split seed is fixed and independent of args.random_state,
    # which only controls the model — changing the hyperparameter
    # does not change which rows land in train vs test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)
    f1    = f1_score(y_test, preds)

    # One metric per line — these exact formats are scraped by the
    # metric_definitions regexes in run_pipeline.py
    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {f1:.4f}")
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
    print("✅ Model saved!")