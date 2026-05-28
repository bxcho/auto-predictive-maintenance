# 🚗 Automotive Predictive Maintenance API

## Overview
This project bridges the gap between Data Science and Software Engineering by deploying a machine learning model as a real-time web service. Using the AI4I 2020 industrial dataset, I engineered a Random Forest classification pipeline to analyze simulated automotive sensor telemetry and predict equipment failure before it happens. 

To make the model production-ready, I wrapped it in a custom FastAPI backend, allowing external systems to send real-time sensor payloads and receive instantaneous risk assessments.

## Business Value
Predictive maintenance minimizes unexpected vehicle downtime, reduces warranty claim costs, and improves overall safety by alerting systems to imminent failures (e.g., Heat Dissipation or Overstrain Failures) rather than relying on scheduled manual checks.

## Tech Stack
* **Machine Learning:** Python, Scikit-Learn (Random Forest), SMOTE (for extreme class imbalance handling)
* **API Deployment:** FastAPI, Uvicorn, Pydantic (Data validation)
* **Data Manipulation:** Pandas, NumPy

## Model Performance & Features
* **Algorithm:** Random Forest Classifier
* **Custom Risk Thresholds:** Engineered a custom 40% probability threshold for failure detection to strictly minimize False Negatives (missing a broken machine).
* **Data Validation:** Implemented strict Pydantic schemas to ensure all incoming telemetry data is correctly formatted before reaching the model.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/bxcho/auto-predictive-maintenance.git](https://github.com/bxcho/auto-predictive-maintenance.git)
   cd auto-predictive-maintenance
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
3. **Train the Model (Generates the .pkl file):**
   ```bash
   python3 train_model.py
4. **Start the FastAPI server:**
   ```bash
   uvicorn app:app --reload
5. **Test the API:** Open your browser and navigate to http://127.0.0.1:8000/docs to use the interactive testing dashboard.