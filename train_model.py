# train_model.py
import os
import json
import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def load_data(csv_path="data/iris.csv"):
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        from sklearn.datasets import load_iris
        data = load_iris()
        X = pd.DataFrame(data.data, columns=['sepal_length','sepal_width','petal_length','petal_width'])
        y = pd.Series(data.target).map(lambda i: data.target_names[i])
        df = pd.concat([X, y.rename('species')], axis=1)
    return df

def main():
    df = load_data()
    # Make sure column names match; expect 'species' as target
    X = df.drop(columns=['species'])
    y = df['species'].astype(str)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    os.makedirs("models", exist_ok=True)
    results = {}
    best_model = None
    best_score = -1
    best_name = None

    for name, model in models.items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "cv_mean": float(np.mean(cv)),
            "cv_std": float(np.std(cv)),
            "test_accuracy": float(test_acc),
            "classification_report": report,
            "confusion_matrix": cm
        }

        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name

    # Save best model (and label encoder so we can decode classes)
    joblib.dump({"model": best_model, "label_encoder": le}, "models/model.pkl")
    with open("models/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved best model: {best_name} (test acc={best_score:.4f})")

if __name__ == "__main__":
    main()
