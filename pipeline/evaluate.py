import pandas as pd
import joblib
import json
import os
import tarfile
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

model_path = "/opt/ml/processing/model/model.tar.gz"
test_path  = "/opt/ml/processing/test/test.csv"
output_dir = "/opt/ml/processing/evaluation"

with tarfile.open(model_path) as tar:
    tar.extractall("/opt/ml/processing/model/")

model = joblib.load("/opt/ml/processing/model/model.joblib")

df    = pd.read_csv(test_path)
X     = df.drop("income_above_50k", axis=1)
y     = df["income_above_50k"]

preds     = model.predict(X)
accuracy  = accuracy_score(y, preds)
precision = precision_score(y, preds)
recall    = recall_score(y, preds)
f1        = f1_score(y, preds)
cm        = confusion_matrix(y, preds, labels=[0, 1])

print(f"✅ Accuracy  : {accuracy:.4f}")
print(f"✅ Precision : {precision:.4f}")
print(f"✅ Recall    : {recall:.4f}")
print(f"✅ F1        : {f1:.4f}")
print(f"✅ Confusion : {cm.tolist()}")

os.makedirs(output_dir, exist_ok=True)

# Two schemas in one file:
#   "metrics"                       → read by the pipeline QualityGate (JsonGet)
#   "binary_classification_metrics" → SageMaker's standard model-quality schema,
#                                     rendered as charts in the Model Registry
report = {
    "version": 0.0,
    "dataset": {
        "item_count": int(len(y))
    },
    "metrics": {
        "accuracy"  : {"value": round(accuracy,  4)},
        "precision" : {"value": round(precision, 4)},
        "recall"    : {"value": round(recall,    4)},
        "f1_score"  : {"value": round(f1,        4)}
    },
    "binary_classification_metrics": {
        "accuracy"  : {"value": round(accuracy,  4), "standard_deviation": "NaN"},
        "precision" : {"value": round(precision, 4), "standard_deviation": "NaN"},
        "recall"    : {"value": round(recall,    4), "standard_deviation": "NaN"},
        "f1"        : {"value": round(f1,        4), "standard_deviation": "NaN"},
        "confusion_matrix": {
            "0": {"0": int(cm[0][0]), "1": int(cm[0][1])},
            "1": {"0": int(cm[1][0]), "1": int(cm[1][1])}
        }
    }
}

with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
    json.dump(report, f, indent=2)

print("✅ Evaluation report saved!")
print(f"✅ Full Report : {json.dumps(report, indent=2)}")