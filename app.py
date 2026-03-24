import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EcoType – Forest Cover Prediction",
    page_icon="🌲",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS for Premium Look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1a472a 0%, #2d6a4f 50%, #40916c 100%);
        padding: 2rem 2rem 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p  { margin: 0.5rem 0 0; opacity: 0.9; font-size: 1rem; }

    .prediction-card {
        background: linear-gradient(135deg, #d8f3dc 0%, #b7e4c7 100%);
        border-left: 6px solid #2d6a4f;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .prediction-card h2 { color: #1b4332; margin: 0 0 0.5rem; font-size: 1.5rem; }
    .prediction-card p  { color: #2d6a4f; margin: 0; font-size: 1.1rem; }

    .info-box {
        background: #f0f7f4;
        border: 1px solid #b7e4c7;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1b4332 0%, #2d6a4f 100%);
    }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2); }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model & Artifacts
# ─────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    path = os.path.join(CURRENT_DIR, "models", "final_random_forest_model.pkl")
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        return None
    return joblib.load(path)


@st.cache_resource
def load_label_encoder():
    path = os.path.join(CURRENT_DIR, "models", "label_encoder.pkl")
    if not os.path.exists(path):
        st.warning("LabelEncoder not found. Using fallback mapping.")
        return None
    return joblib.load(path)


@st.cache_resource
def load_selected_features():
    path = os.path.join(CURRENT_DIR, "models", "selected_features.pkl")
    if not os.path.exists(path):
        return None
    return joblib.load(path)


model = load_model()
label_encoder = load_label_encoder()
selected_features = load_selected_features()

# Fallback cover type mapping (used only if encoder is unavailable)
COVER_TYPE_FALLBACK = {
    0: "Aspen", 1: "Cottonwood/Willow", 2: "Douglas-fir",
    3: "Krummholz", 4: "Lodgepole Pine", 5: "Ponderosa Pine", 6: "Spruce/Fir"
}

# Cover type descriptions
COVER_DESCRIPTIONS = {
    "Spruce/Fir": "🌲 Coniferous forests found at high elevations with cool, moist climates.",
    "Lodgepole Pine": "🏔️ Hardy pine species common in mountain environments.",
    "Ponderosa Pine": "🌿 Large pine trees found at lower, drier elevations.",
    "Cottonwood/Willow": "💧 Deciduous trees found near streams and low-elevation riparian zones.",
    "Aspen": "🍂 Deciduous trees with iconic white bark, found in moist mountain areas.",
    "Douglas-fir": "🌲 Tall, commercially important conifers found at mid-elevations.",
    "Krummholz": "⛰️ Stunted, wind-shaped trees found at the treeline of alpine zones."
}

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌲 EcoType")
    st.markdown("**Forest Cover Type Prediction**")
    st.markdown("---")
    st.markdown("#### 📌 About")
    st.markdown(
        "Predict forest cover type using cartographic and environmental features. "
        "Powered by a tuned Random Forest classifier trained on 145K+ data points."
    )
    st.markdown("---")
    st.markdown("#### 🌍 Cover Types")
    for name, desc in COVER_DESCRIPTIONS.items():
        st.markdown(f"**{name}**  \n{desc}")
    st.markdown("---")
    st.markdown("#### 🛠️ Tech Stack")
    st.markdown("Python • Scikit-learn • XGBoost • Streamlit")
    st.markdown("---")
    st.markdown("👤 **Abishek Murugan**")

# ─────────────────────────────────────────────
# Main Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🌲 EcoType: Forest Cover Type Prediction</h1>
    <p>Predict forest cover type using cartographic and environmental features</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# User Inputs
# ─────────────────────────────────────────────
st.markdown("### 📥 Enter Feature Values")
st.markdown('<div class="info-box">Provide the cartographic measurements for the area you want to classify. '
            'All distance values are in meters.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("##### 🏔️ Terrain Features")
    Elevation = st.number_input("Elevation (meters)", min_value=0, max_value=5000, value=2500,
                                 help="Height above sea level in meters")
    Aspect = st.number_input("Aspect (degrees)", min_value=0, max_value=360, value=180,
                              help="Direction the slope faces (0-360°)")
    Slope = st.number_input("Slope (degrees)", min_value=0, max_value=60, value=10,
                             help="Steepness of the terrain")
    Wilderness_Area = st.selectbox("Wilderness Area", [1, 2, 3, 4],
                                    help="Designated wilderness area ID")
    Soil_Type = st.selectbox("Soil Type", list(range(1, 41)),
                              help="Soil classification type (1-40)")

with col2:
    st.markdown("##### 📏 Distance Features")
    Horizontal_Distance_To_Hydrology = st.number_input(
        "Horizontal Dist. to Hydrology (m)", min_value=0, max_value=2000, value=250,
        help="Horizontal distance to nearest water source")
    Vertical_Distance_To_Hydrology = st.number_input(
        "Vertical Dist. to Hydrology (m)", min_value=-500, max_value=500, value=0,
        help="Vertical distance to nearest water source")
    Horizontal_Distance_To_Roadways = st.number_input(
        "Horizontal Dist. to Roadways (m)", min_value=0, max_value=10000, value=2000,
        help="Horizontal distance to nearest road")
    Horizontal_Distance_To_Fire_Points = st.number_input(
        "Horizontal Dist. to Fire Points (m)", min_value=0, max_value=10000, value=3000,
        help="Horizontal distance to nearest wildfire ignition point")

st.markdown("---")
st.markdown("### ☀️ Hillshade Indices")
st.markdown('<div class="info-box">Illumination indices ranging from 0 (dark) to 255 (bright) '
            'at different times of day.</div>', unsafe_allow_html=True)

h_col1, h_col2, h_col3 = st.columns(3)
with h_col1:
    Hillshade_9am = st.slider("Hillshade 9am", 0, 255, 200)
with h_col2:
    Hillshade_Noon = st.slider("Hillshade Noon", 0, 255, 220)
with h_col3:
    Hillshade_3pm = st.slider("Hillshade 3pm", 0, 255, 180)

# ─────────────────────────────────────────────
# Compute Engineered Features
# ─────────────────────────────────────────────
Hillshade_mean = np.mean([Hillshade_9am, Hillshade_Noon, Hillshade_3pm])
Hillshade_diff_9am_3pm = Hillshade_9am - Hillshade_3pm
Elevation_Slope = Elevation * Slope
Hydrology_Distance_Mag = (Horizontal_Distance_To_Hydrology**2 +
                          Vertical_Distance_To_Hydrology**2) ** 0.5
Road_Fire_Distance_Ratio = Horizontal_Distance_To_Roadways / (Horizontal_Distance_To_Fire_Points + 1)

# ─────────────────────────────────────────────
# Build Input DataFrame
# ─────────────────────────────────────────────
# Full feature set (order must match the training feature order)
ALL_FEATURES = {
    'Elevation': Elevation,
    'Aspect': Aspect,
    'Slope': Slope,
    'Horizontal_Distance_To_Hydrology': Horizontal_Distance_To_Hydrology,
    'Vertical_Distance_To_Hydrology': Vertical_Distance_To_Hydrology,
    'Horizontal_Distance_To_Roadways': Horizontal_Distance_To_Roadways,
    'Hillshade_9am': Hillshade_9am,
    'Hillshade_Noon': Hillshade_Noon,
    'Hillshade_3pm': Hillshade_3pm,
    'Horizontal_Distance_To_Fire_Points': Horizontal_Distance_To_Fire_Points,
    'Wilderness_Area': Wilderness_Area,
    'Soil_Type': Soil_Type,
    'Hillshade_diff_9am_3pm': Hillshade_diff_9am_3pm,
    'Hillshade_mean': Hillshade_mean,
    'Elevation_Slope': Elevation_Slope,
    'Hydrology_Distance_Mag': Hydrology_Distance_Mag,
    'Road_Fire_Distance_Ratio': Road_Fire_Distance_Ratio,
}

input_df = pd.DataFrame([ALL_FEATURES])

# Use only selected features if available
if selected_features is not None:
    input_df = input_df[selected_features]

# ─────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────
st.markdown("---")
predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
with predict_col2:
    predict_btn = st.button("🔍 Predict Forest Cover Type", use_container_width=True, type="primary")

if predict_btn:
    if model is not None:
        try:
            prediction = model.predict(input_df)[0]

            # Inverse transform using saved LabelEncoder
            if label_encoder is not None:
                cover_type_name = label_encoder.inverse_transform([prediction])[0]
            else:
                cover_type_name = COVER_TYPE_FALLBACK.get(prediction, f"Unknown ({prediction})")

            # Display prediction with confidence
            probabilities = model.predict_proba(input_df)[0]
            confidence = probabilities[prediction] * 100

            st.markdown(f"""
            <div class="prediction-card">
                <h2>🌿 Predicted Forest Cover Type</h2>
                <p style="font-size: 1.8rem; font-weight: 700; color: #1b4332;">{cover_type_name}</p>
                <p>Prediction Confidence: <strong>{confidence:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            # Show description
            if cover_type_name in COVER_DESCRIPTIONS:
                st.info(COVER_DESCRIPTIONS[cover_type_name])

            # Show probability distribution
            st.markdown("#### 📊 Prediction Probabilities")
            if label_encoder is not None:
                prob_df = pd.DataFrame({
                    'Cover Type': label_encoder.classes_,
                    'Probability': probabilities
                }).sort_values('Probability', ascending=True)
            else:
                prob_df = pd.DataFrame({
                    'Cover Type': [COVER_TYPE_FALLBACK.get(i, f"Class {i}") for i in range(len(probabilities))],
                    'Probability': probabilities
                }).sort_values('Probability', ascending=True)

            st.bar_chart(prob_df.set_index('Cover Type'), horizontal=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("⚠️ Model not loaded. Please ensure model files exist in the `models/` directory.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#6c757d; font-size:0.85rem;'>"
    "EcoType v2.0 • Built with Streamlit • Abishek Murugan"
    "</div>",
    unsafe_allow_html=True
)
