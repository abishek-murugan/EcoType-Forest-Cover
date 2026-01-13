# src/config.py

# ===============================
# Paths
# ===============================
DATA_PATH = "/home/abishek-murugan/Project Folder/GUVI Projects/EcoType-Forest-Cover/data/processed/feature_engineered_forest_cover_data.csv"

MODEL_PATH = "models/final_random_forest_model.pkl"

TARGET_COLUMN = "Cover_Type"
# ===============================
# Train / Test Split
# ===============================
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===============================
# Random Forest (Final Tuned Params)
# ===============================
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 30,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "random_state": RANDOM_STATE,
    "n_jobs": 2
}
