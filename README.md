# 🌍 Air Quality Index (AQI) Prediction Using Machine Learning

### 🤖 Machine Learning Based Project using Python, Scikit-learn & Streamlit

An end-to-end Machine Learning project that predicts the Air Quality Index (AQI) of different Indian cities using air pollutant concentration data.

This project follows a complete Machine Learning workflow, including data collection, data preprocessing, Exploratory Data Analysis (EDA), model training, model evaluation, error investigation, and interactive dashboard development using Streamlit.

---

## 🎯 Project Objective

The main objective of this project is to develop a Machine Learning model that predicts the Air Quality Index (AQI) based on different air pollutant concentrations.

### Key Objectives

- Analyze air quality data from 50 Indian cities.
- Perform data cleaning and preprocessing.
- Explore relationships between air pollutants and AQI.
- Visualize AQI distribution and pollution trends.
- Train and compare multiple Machine Learning regression models.
- Evaluate model performance using different metrics.
- Identify important features affecting model predictions.
- Investigate prediction errors and data distribution.
- Develop an interactive Streamlit application for AQI prediction and analysis.

---

## 🛠️ Technologies & Tools

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
  <img src="https://img.shields.io/badge/VS_Code-007ACC?style=for-the-badge&logo=visualstudiocode&logoColor=white"/>
</p>

### Main Tools Used

- 🐍 **Python** – Programming and Machine Learning implementation
- 🐼 **Pandas** – Data cleaning and manipulation
- 🔢 **NumPy** – Numerical operations
- 📊 **Matplotlib** – Data visualization
- 📈 **Seaborn** – Statistical data visualization
- 🤖 **Scikit-learn** – Machine Learning model development
- 🌐 **Streamlit** – Interactive web application development
- 💾 **Joblib** – Saving and loading trained models
- 💻 **VS Code** – Project development environment
- 🔧 **Git & GitHub** – Version control and project hosting

---

## 📂 Dataset Information

**Dataset Name:** India Daily Weather and Air Quality

**Dataset Source:** Kaggle

**Dataset Creator:** AkshayKarthick

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

**License Link:** https://creativecommons.org/licenses/by/4.0/

### Dataset Details

| Description | Details |
|---|---|
| Total Records | 77,400 |
| Total Cities | 50 |
| Date Coverage | August 2022 – October 2026 |
| Original Columns | 19 |
| Cleaned Columns | 9 |
| Target Variable | India AQI |

### Features Used

**Input Features:**
- PM2.5 – Fine Particulate Matter
- PM10 – Particulate Matter
- NO2 – Nitrogen Dioxide
- SO2 – Sulphur Dioxide
- O3 – Ozone
- CO – Carbon Monoxide
- City – City name

**Target Variable:** India AQI

The original dataset was cleaned and preprocessed before being used for Machine Learning analysis.

---

## ⚙️ Project Workflow

### 1. Dataset Understanding
- Understanding dataset structure and columns.
- Checking missing values and duplicate records.
- Identifying input features and target variable.

### 2. Data Preprocessing
- Selecting required columns.
- Converting date column into datetime format.
- Handling missing and duplicate records.
- Checking invalid pollutant and AQI values.
- Saving the cleaned dataset.

### 3. Exploratory Data Analysis (EDA)
- AQI Distribution Analysis
- Top 10 Cities by Average AQI
- Monthly AQI Trend Analysis
- Correlation Heatmap
- PM2.5 vs AQI Relationship

### 4. Model Training
Implemented and compared three regression algorithms.

### 5. Model Evaluation
Evaluated model performance using MAE, MSE, RMSE and R² Score.

### 6. Error Investigation
- City-wise prediction error analysis.
- AQI range-wise performance analysis.
- High AQI prediction investigation.
- Training and testing data distribution comparison.

### 7. Model Improvement
Compared different Random Forest configurations and selected the best-performing validation model.

### 8. Final Testing
Evaluated the selected model on the chronological testing dataset.

### 9. Streamlit Application
Developed an interactive web application for AQI prediction and historical dataset analysis.

---

## 🤖 Machine Learning Algorithms

The following regression algorithms were implemented:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

Random Forest Regressor was selected based on model comparison and validation results.

---

## 📊 Model Performance

The final selected Random Forest model was evaluated on 15,500 testing records.

| Evaluation Metrics | Result |
|---|---:|
| MAE | 0.639 |
| MSE | 57.727 |
| RMSE | 7.598 |
| R² Score | 0.9884 |

**Note:** Overall evaluation metrics do not indicate uniform performance across all AQI ranges. The model has higher prediction errors for certain extreme AQI observations, particularly in the 401–500 range.

---

## 📈 Data Visualization

The project includes the following visualizations:

- AQI Distribution
- Top 10 Cities by Average AQI
- Monthly AQI Trends
- Correlation Heatmap
- PM2.5 vs AQI Scatter Plot
- Feature Importance Graph
- Actual vs Predicted AQI
- Prediction Error Distribution
- PM10 Distribution Comparison
- AQI Distribution Comparison
- PM10 vs AQI Relationship
- High AQI Prediction Analysis

---

## 🌐 Streamlit Web Application

An interactive web application was developed using Streamlit.

### Dashboard Overview

- Total Dataset Records
- Total Cities
- Average AQI
- Maximum AQI
- City-wise Average AQI
- AQI Distribution

### AQI Prediction

- Select a city.
- Enter pollutant concentration values.
- Predict AQI using the trained Random Forest model.
- Display predicted AQI and its category.
- Visualize pollutant values.

### AQI Categories

| AQI Range | Category |
|---|---|
| 0–50 | Good |
| 51–100 | Satisfactory |
| 101–200 | Moderate |
| 201–300 | Poor |
| 301–400 | Very Poor |
| 401–500 | Severe |

---

## 🖥️ Application Screenshots

### 1. Dashboard Overview

![Dashboard Overview](Dashboard_Screenshot\Dashboard.png)

### 2. AQI Prediction

![AQI Prediction](Dashboard_Screenshot\AQI_Distribution.png)

### 3. Pollutant Analysis

![Pollutant Analysis](Dashboard_Screenshot\Pollutant_Analysis.png)

### 4. City Wise Prediction

![City Wise Prediction](Dashboard_Screenshot\City_Wise_Prediction.png)

---

AQI_Prediction/
│
├── dataset/
│   ├── air_quality_daily.csv
│   └── cleaned_aqi.csv
│
├── Dashboard_Screenshot/
│   ├── AQI_Distribution.png
│   ├── City_Wise_Prediction.png
│   ├── Dashboard.png
│   └── Pollutant_Analysis.png
│
├── notebooks/
│   ├── 01_dataset_understanding.py
│   ├── 02_data_preprocessing.py
│   ├── 03_eda.py
│   ├── 04_model_training.py
│   ├── 05_model_evaluation.py
│   ├── 06_error_investigation.py
│   ├── 07_data_distribution_check.py
│   ├── 08_distribution_visualization.py
│   ├── 09_model_improvement.py
│   └── 10_final_testing.py
│
├── models/
│   ├── best_aqi_model.pkl
│   ├── improved_aqi_model.pkl
│   ├── aqi_predictions.csv
│   ├── final_aqi_predictions.csv
│   ├── model_comparison.csv
│   └── improvement_comparison.csv
│
├── graphs/
│   └── Visualization Graphs
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

## 🚀 Installation & Execution

### 1. Navigate to Project Folder

```bash
cd AQI_Prediction
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser:

```text
http://localhost:8501
```

## 👩‍💻 Author

**Isha Namdev Jadhav**

B.Tech – Computer Science and Engineering

