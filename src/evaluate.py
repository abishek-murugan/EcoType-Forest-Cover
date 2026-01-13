import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATA_PATH, MODEL_PATH, TARGET_COLUMN


def evaluate():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))


if __name__ == "__main__":
    evaluate()
