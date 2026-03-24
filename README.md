# 🌲 EcoType: Forest Cover Type Prediction

## 📌 Project Overview

EcoType is a **machine learning classification system** that predicts the **type of forest cover** in a given geographical area using cartographic and environmental features such as elevation, slope, soil type, wilderness area, hydrology distances, and hillshade indices.

This project supports **environmental monitoring, forest resource management, and land-use planning** by providing automated, reliable forest cover type identification.

---

## 🎯 Problem Statement

Build a classification model that predicts one of **7 forest cover types** based on numerical and encoded categorical features derived from geographical data.

---

## 📊 Dataset Information

* **Source:** [Forest Cover Type Dataset](https://www.kaggle.com/datasets)
* **Rows:** 145,890
* **Columns:** 13 original + 5 engineered = 18 features
* **Target Variable:** `Cover_Type` (7 classes)

### Cover Type Classes

| Label | Forest Cover Type   | Description |
|-------|---------------------|-------------|
| 0     | Aspen               | Deciduous trees with white bark, moist mountain areas |
| 1     | Cottonwood/Willow   | Deciduous trees near streams, low-elevation zones |
| 2     | Douglas-fir         | Tall conifers at mid-elevations |
| 3     | Krummholz           | Stunted alpine treeline trees |
| 4     | Lodgepole Pine      | Hardy mountain pine species |
| 5     | Ponderosa Pine      | Large pines at lower, drier elevations |
| 6     | Spruce/Fir          | Coniferous forests at high elevations |

---

## ⚙️ Project Workflow

### 1. Data Understanding & Cleaning (`1_data_understanding.ipynb`)
- Dataset exploration (shape, info, describe, value_counts)
- Missing value and duplicate checks
- Outlier detection using IQR method
- Outlier treatment via IQR-based capping
- Skewness analysis

### 2. Exploratory Data Analysis (`2_eda.ipynb`)
- Univariate analysis (histograms, distributions)
- Bivariate analysis (boxplots, violin plots by cover type)
- Correlation heatmap with annotations
- Pairplot of key features
- Baseline feature importance visualization

### 3. Feature Engineering (`3_feature_engineeing.ipynb`)
- Created 5 engineered features:
  - `Hillshade_diff_9am_3pm` – Sunlight direction change
  - `Hillshade_mean` – Average illumination
  - `Elevation_Slope` – Terrain interaction
  - `Hydrology_Distance_Mag` – True distance to water
  - `Road_Fire_Distance_Ratio` – Relative accessibility
- Label encoding of target variable
- **Saved LabelEncoder** for Streamlit inverse transform

### 4. Model Building & Evaluation (`4_modeling.ipynb`)
- Feature selection using importance threshold
- Class imbalance handling with SMOTE (training data only)
- **5 classification models built and compared:**
  - ✅ Random Forest
  - ✅ Decision Tree
  - ✅ Logistic Regression
  - ✅ K-Nearest Neighbors (KNN)
  - ✅ XGBoost
- Model comparison table and bar chart
- Confusion matrices for all models
- 5-fold cross-validation
- Hyperparameter tuning with RandomizedSearchCV
- Best model saved as `.pkl`

### 5. Deployment (`app.py`)
- Interactive **Streamlit web application**
- Real-time prediction using trained model
- **Inverse transform** of target using saved LabelEncoder
- Prediction confidence and probability distribution display
- Premium UI with custom styling

---

## 🧠 Final Model

* **Algorithm:** Random Forest Classifier (Tuned)
* **Tuning:** RandomizedSearchCV with 3-fold CV, 20 iterations
* **Saved Artifacts:** Model, selected features, LabelEncoder, SMOTE object

---

## 📂 Project Structure

```text
ECOTYPE-FOREST-COVER/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
│       ├── capped_forest_cover_data.csv
│       └── feature_engineered_forest_cover_data.csv
│
├── models/
│   ├── final_random_forest_model.pkl
│   ├── selected_features.pkl
│   ├── label_encoder.pkl
│   └── smote.pkl
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_feature_engineeing.ipynb
│   └── 4_modeling.ipynb
│
├── app.py
├── README.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Notebooks (in order)

```bash
cd notebooks/
jupyter notebook
# Run notebooks 1 → 2 → 3 → 4 sequentially
```

### 3️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.x |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost |
| Imbalance | imbalanced-learn (SMOTE) |
| Deployment | Streamlit |
| Serialization | Joblib |

---

## ✅ Key Highlights

* End-to-end ML pipeline from data cleaning to deployment
* **5 models compared** with accuracy metrics and confusion matrices
* Feature-engineered dataset with 5 derived features
* Class imbalance handled with SMOTE (applied correctly on training data only)
* Hyperparameter tuning with RandomizedSearchCV + cross-validation
* LabelEncoder saved and used for **inverse transform** in Streamlit
* Production-quality Streamlit app with premium UI
* All notebooks use **relative paths** for portability
* Clean, well-documented codebase with markdown explanations

---

## 👤 Author

**Abishek Murugan**

---

## 🎉 Status

✅ Project Completed  
🚀 Ready for Evaluation