# 🌲 EcoType: Forest Cover Type Prediction

## 📌 Project Overview

EcoType is a machine learning–based classification system designed to predict the **forest cover type** of a geographical area using cartographic and environmental features such as elevation, slope, hydrology distances, hillshade indices, soil type, and wilderness area indicators.

This project helps in **environmental monitoring, forest resource management, and land-use planning** by providing accurate and automated forest cover identification.

---

## 🎯 Problem Statement

To build a classification model that predicts one of **7 forest cover types** based on numerical and encoded categorical features derived from geographical data.

---

## 📊 Dataset Information

* **Source:** Forest Cover Type Dataset
* **Rows:** 145,891
* **Columns:** 13 (after preprocessing & feature engineering)
* **Target Variable:** `Cover_Type` (7 classes)

### Cover Type Classes

| Label | Forest Cover Type   |
| ----- | ------------------- |
| 1     | Spruce / Fir        |
| 2     | Lodgepole Pine      |
| 3     | Ponderosa Pine      |
| 4     | Cottonwood / Willow |
| 5     | Aspen               |
| 6     | Douglas-fir         |
| 7     | Krummholz           |

---

## ⚙️ Project Workflow

1. **Data Understanding & EDA**

   * Dataset exploration
   * Feature distribution analysis
   * Class imbalance analysis

2. **Data Preprocessing**

   * Handling missing values
   * Outlier treatment
   * Feature scaling and transformation

3. **Feature Engineering**

   * Road–Fire distance ratio
   * Hydrology distance magnitude
   * Hillshade statistics (mean, difference)
   * Elevation–Slope interaction feature

4. **Model Building**

   * Random Forest
   * Decision Tree
   * Logistic Regression
   * K-Nearest Neighbors
   * XGBoost

5. **Model Evaluation**

   * Accuracy
   * Confusion Matrix
   * Classification Report

6. **Final Model Selection**

   * Best-performing model selected based on evaluation metrics
   * Model saved as `.pkl` file

7. **Deployment**

   * Interactive **Streamlit web application**
   * Real-time prediction using trained model

---

## 🧠 Final Model

* **Algorithm:** Random Forest Classifier
* **Saved Model:** `final_random_forest_model.pkl`
* **Reason for Selection:** High accuracy, robustness to non-linear features, and strong generalization

---

## 🖥️ Streamlit Web Application

The Streamlit app allows users to:

* Enter environmental and cartographic values
* Predict the forest cover type instantly
* View human-readable prediction results

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
│   └── smote.pkl
│
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   ├── 3_feature_engineineg.ipynb
│   ├── 4_modeling.ipynb
│   └── final_model.ipynb
│
├── app.py
├── README.md
├── requirements.txt
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Run Streamlit App

```bash
streamlit run app.py
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib

---

## ✅ Key Highlights

* End-to-end ML pipeline
* Feature-engineered dataset
* Multiple model comparison
* Clean, interactive deployment
* Production-ready inference flow

---

## 📌 Conclusion

EcoType demonstrates how machine learning can be effectively applied to environmental and geospatial data to support decision-making in forestry and ecological research.

---

## 👤 Author

**Abishek Murugan**

---

## 🎉 Status

✅ Project Completed
🚀 Ready for Evaluation

---