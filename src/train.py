import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from config import (
    DATA_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    RF_PARAMS
)


def train():
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COLUMN])
    Y = df[TARGET_COLUMN]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=Y
    )

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, Y_train)

    joblib.dump(model, MODEL_PATH)
    print("✅ Model trained and saved successfully")


if __name__ == "__main__":
    train()
