import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="EcoType – Forest Cover Prediction",
    layout="centered"
)

st.title("🌲 EcoType: Forest Cover Type Prediction")
st.write("Predict forest cover type using cartographic and environmental features.")

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("/home/abishek-murugan/Project Folder/GUVI Projects/EcoType-Forest-Cover/models/final_random_forest_model.pkl")

model = load_model()

# -------------------------------
# Feature order (DO NOT CHANGE)
# -------------------------------
FEATURE_ORDER = [
    'Elevation',
    'Horizontal_Distance_To_Roadways',
    'Horizontal_Distance_To_Fire_Points',
    'Road_Fire_Distance_Ratio',
    'Soil_Type',
    'Wilderness_Area',
    'Hydrology_Distance_Mag',
    'Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Hydrology',
    'Aspect',
    'Elevation_Slope',
    'Hillshade_9am',
    'Hillshade_diff_9am_3pm',
    'Hillshade_Noon',
    'Hillshade_mean',
    'Hillshade_3pm',
    'Slope'
]

# -------------------------------
# Cover type mapping
# -------------------------------
COVER_TYPE_MAP = {
    1: "Spruce / Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood / Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("📥 Enter Feature Values")

Elevation = st.number_input("Elevation (meters)", 0, 5000, 2500)
Horizontal_Distance_To_Roadways = st.number_input("Horizontal Distance to Roadways (m)", 0, 50000, 2000)
Horizontal_Distance_To_Fire_Points = st.number_input("Horizontal Distance to Fire Points (m)", 0, 50000, 3000)

Road_Fire_Distance_Ratio = st.number_input("Road–Fire Distance Ratio", 0.0, 50.0, 1.0)

Soil_Type = st.selectbox("Soil Type (encoded)", list(range(1, 41)))
Wilderness_Area = st.selectbox("Wilderness Area (encoded)", [1, 2, 3, 4])

Hydrology_Distance_Mag = st.number_input("Hydrology Distance Magnitude", 0.0, 10000.0, 500.0)
Vertical_Distance_To_Hydrology = st.number_input("Vertical Distance to Hydrology", -500, 500, 0)
Horizontal_Distance_To_Hydrology = st.number_input("Horizontal Distance to Hydrology", 0, 50000, 1000)

Aspect = st.number_input("Aspect (degrees)", 0, 360, 180)
Slope = st.number_input("Slope (degrees)", 0, 60, 10)

Elevation_Slope = Elevation * Slope

Hillshade_9am = st.slider("Hillshade 9am", 0, 255, 200)
Hillshade_Noon = st.slider("Hillshade Noon", 0, 255, 220)
Hillshade_3pm = st.slider("Hillshade 3pm", 0, 255, 180)

Hillshade_mean = np.mean([Hillshade_9am, Hillshade_Noon, Hillshade_3pm])
Hillshade_diff_9am_3pm = Hillshade_9am - Hillshade_3pm

# -------------------------------
# Create input DataFrame
# -------------------------------
input_data = {
    'Elevation': Elevation,
    'Horizontal_Distance_To_Roadways': Horizontal_Distance_To_Roadways,
    'Horizontal_Distance_To_Fire_Points': Horizontal_Distance_To_Fire_Points,
    'Road_Fire_Distance_Ratio': Road_Fire_Distance_Ratio,
    'Soil_Type': Soil_Type,
    'Wilderness_Area': Wilderness_Area,
    'Hydrology_Distance_Mag': Hydrology_Distance_Mag,
    'Vertical_Distance_To_Hydrology': Vertical_Distance_To_Hydrology,
    'Horizontal_Distance_To_Hydrology': Horizontal_Distance_To_Hydrology,
    'Aspect': Aspect,
    'Elevation_Slope': Elevation_Slope,
    'Hillshade_9am': Hillshade_9am,
    'Hillshade_diff_9am_3pm': Hillshade_diff_9am_3pm,
    'Hillshade_Noon': Hillshade_Noon,
    'Hillshade_mean': Hillshade_mean,
    'Hillshade_3pm': Hillshade_3pm,
    'Slope': Slope
}

input_df = pd.DataFrame([input_data])

# Reorder columns EXACTLY as training
input_df = input_df[FEATURE_ORDER]

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔍 Predict Forest Cover Type"):
    prediction = model.predict(input_df)[0]
    cover_type = COVER_TYPE_MAP.get(prediction, "Unknown")

    st.success(f"🌿 **Predicted Forest Cover Type:** {cover_type}")
