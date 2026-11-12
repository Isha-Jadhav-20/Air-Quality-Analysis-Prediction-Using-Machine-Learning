# AIR QUALITY INDEX (AQI) PREDICTION USING MACHINE LEARNING

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# 1) PAGE CONFIGURATION

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# 2) CUSTOM CSS DESIGN

st.markdown("""
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #071b2b,
        #102f42,
        #071b2b
    );
    color: white;
}

h1, h2, h3 {
    color: #00d4ff !important;
}

p, label {
    color: #e0f7fa !important;
}

div[data-testid="stMetric"] {
    background: #12394b;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #00d4ff;
}

div[data-testid="stMetric"] label {
    color: #aeeaf5 !important;
}

div[data-testid="stMetricValue"] {
    color: #ffffff !important;
}

.stButton button {
    background: linear-gradient(
        90deg,
        #00bcd4,
        #007c91
    );
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 25px;
    font-weight: bold;
}

.stButton button:hover {
    background: #00d4ff;
    color: black;
}

div[data-baseweb="select"] {
    background-color: #12394b;
}

</style>
""", unsafe_allow_html=True)


# 3) PROJECT FILE PATHS

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "improved_aqi_model.pkl"

DATA_PATH = BASE_DIR / "dataset" / "cleaned_aqi.csv"

# 4. LOAD MODEL AND DATASET

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model
@st.cache_data
def load_dataset():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data
try:
    model = load_model()
    df = load_dataset()
except FileNotFoundError as error:
    st.error(f"File not found: {error}")
    st.stop()

# 5) PROJECT TITLE

st.title("🌍 Air Quality Index (AQI) Prediction")

st.write(
    "Machine Learning Based Air Quality Monitoring "
    "and Prediction System"
)
st.divider()

# 6) SIDEBAR NAVIGATION

st.sidebar.title("🌍 AQI System")
st.sidebar.write("Navigation")
page = st.sidebar.radio(
    "Select Section",
    [
        "Dashboard Overview",
        "AQI Prediction"
    ]
)
st.sidebar.divider()
st.sidebar.info(
    "This application uses a trained "
    "Machine Learning model to predict AQI."
)

# 7) AQI CATEGORY FUNCTION

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "#00c853"
    elif aqi <= 100:
        return "Satisfactory", "#8bc34a"
    elif aqi <= 200:
        return "Moderate", "#ffeb3b"
    elif aqi <= 300:
        return "Poor", "#ff9800"
    elif aqi <= 400:
        return "Very Poor", "#f44336"
    else:
        return "Severe", "#9c27b0"

# 8) DASHBOARD OVERVIEW

if page == "Dashboard Overview":
    st.header("📊 Dashboard Overview")
    st.write(
        "Explore historical air quality information "
        "from the available dataset."
    )
    st.divider()
    # Calculate Dashboard Information
    total_records = len(df)
    total_cities = df["city"].nunique()
    average_aqi = df["india_aqi"].mean()
    maximum_aqi = df["india_aqi"].max()
    # Display Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Records",
            f"{total_records:,}"
        )
    with col2:
        st.metric(
            "Total Cities",
            total_cities
        )
    with col3:
        st.metric(
            "Average AQI",
            round(average_aqi, 2)
        )
    with col4:
        st.metric(
            "Maximum AQI",
            round(maximum_aqi, 2)
        )
    st.divider()
    # City-wise Average AQI

    st.subheader("🏙️ Top 10 Cities by Average AQI")
    city_aqi = (
        df.groupby("city")["india_aqi"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    st.bar_chart(
        city_aqi,
        color="#00d4ff"
    )
    st.divider()
    # AQI Distribution
    st.subheader("🌫️ AQI Distribution")
    aqi_ranges = pd.cut(
        df["india_aqi"],
        bins=[0, 50, 100, 200, 300, 400, 500],
        labels=[
            "Good",
            "Satisfactory",
            "Moderate",
            "Poor",
            "Very Poor",
            "Severe"
        ],
        include_lowest=True
    )
    aqi_counts = aqi_ranges.value_counts().sort_index()
    st.bar_chart(
        aqi_counts,
        color="#00bcd4"
    )
    st.divider()
    # Dataset Information

    st.subheader("📁 Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
         st.write("**Dataset Start Date:**")
         st.write(df["date"].min().strftime("%d-%m-%Y"))
    with col2:
        st.write("**Dataset End Date:**")
        st.write(df["date"].max().strftime("%d-%m-%Y"))
    st.caption(
        "Dashboard values are calculated from historical dataset records. "
        "They are not live AQI readings."
    )

# 9) AQI PREDICTION SECTION

elif page == "AQI Prediction":
    st.header("🌫️ AQI Prediction")
    st.write(
        "Enter pollutant values to predict "
        "the Air Quality Index."
    )
    st.divider()
    # City Selection

    city_list = sorted(df["city"].unique())
    selected_city = st.selectbox(
        "Select City",
        city_list
    )
    # Default pollutant values

    city_data = df[df["city"] == selected_city]
    pm25_default = float(city_data["pm2_5_mean"].median())
    pm10_default = float(city_data["pm10_mean"].median())
    no2_default = float(city_data["no2_mean"].median())
    so2_default = float(city_data["so2_mean"].median())
    o3_default = float(city_data["o3_8h_max"].median())
    co_default = float(city_data["co_8h_max"].median())

    # User Input Form

    st.subheader("Enter Pollution Details")
    with st.form("aqi_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pm25 = st.number_input(
                "PM2.5",
                min_value=0.0,
                max_value=2000.0,
                value=pm25_default
            )

            pm10 = st.number_input(
                "PM10",
                min_value=0.0,
                max_value=2000.0,
                value=pm10_default
            )

            no2 = st.number_input(
                "NO2",
                min_value=0.0,
                max_value=2000.0,
                value=no2_default
            )


        with col2:

            so2 = st.number_input(
                "SO2",
                min_value=0.0,
                max_value=2000.0,
                value=so2_default
            )

            o3 = st.number_input(
                "O3",
                min_value=0.0,
                max_value=2000.0,
                value=o3_default
            )

            co = st.number_input(
                "CO",
                min_value=0.0,
                max_value=2000.0,
                value=co_default
            )


        predict_button = st.form_submit_button(
            "🔍 Predict AQI",
            use_container_width=True
        )

    # 10) MACHINE LEARNING PREDICTION

    if predict_button:
        input_data = pd.DataFrame({
            "pm2_5_mean": [pm25],
            "pm10_mean": [pm10],
            "no2_mean": [no2],
            "so2_mean": [so2],
            "o3_8h_max": [o3],
            "co_8h_max": [co],
            "city": [selected_city]
        })

        # Predict AQI

        prediction = model.predict(input_data)[0]
        prediction = max(0, min(500, prediction))
        category, category_color = get_aqi_category(prediction)
        st.divider()
        st.subheader("🎯 AQI Prediction Result")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Predicted AQI",
                round(prediction, 2)
            )
        with col2:
            st.markdown(
                f"""
                <div style="
                    background-color:{category_color};
                    padding:20px;
                    border-radius:12px;
                    text-align:center;
                    color:black;
                    font-size:22px;
                    font-weight:bold;
                ">
                    {category}
                </div>
                """,
                unsafe_allow_html=True
            )
        st.divider()
        # Pollutant Analysis Chart
        st.subheader("📊 Pollutant Analysis")
        pollutant_data = pd.DataFrame({
            "Pollutant": [
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "O3",
                "CO"
            ],

            "Value": [
                pm25,
                pm10,
                no2,
                so2,
                o3,
                co
            ]

        })

        st.bar_chart(
            pollutant_data.set_index("Pollutant"),
            color="#00d4ff"
        )

        st.divider()

        # AQI Reference Table
        st.subheader("📋 AQI Category Reference")
        aqi_reference = pd.DataFrame({

            "AQI Range": [
                "0 - 50",
                "51 - 100",
                "101 - 200",
                "201 - 300",
                "301 - 400",
                "401 - 500"
            ],

            "Category": [
                "Good",
                "Satisfactory",
                "Moderate",
                "Poor",
                "Very Poor",
                "Severe"
            ]

        })

        st.dataframe(
            aqi_reference,
            use_container_width=True,
            hide_index=True
        )
