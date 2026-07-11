# VitalSense Ultra 2.0 — Multi-Disease Risk Prediction Dashboard

An interactive Streamlit dashboard that predicts individual risk scores for **Diabetes, Hypertension, and Heart Disease** using a calibrated ensemble of Gradient Boosting and Random Forest classifiers, with rich visual analytics for exploring risk factors across a population cohort.

## Features
- Real-time risk scoring across three conditions using a calibrated `VotingClassifier` ensemble
- Interactive data explorer: sleep distribution, activity vs. diabetes risk, feature correlation heatmap, risk-by-diet-quality box plots, and population-level risk distributions
- Custom dark-themed UI built with Streamlit + Plotly
- Cohort-based comparison — see how your inputs compare against a simulated population sample

## Tech Stack
- **Frontend/App:** Streamlit
- **ML Models:** scikit-learn (Gradient Boosting, Random Forest, Voting Classifier, Calibrated Classifier)
- **Data Processing:** Pandas, NumPy
- **Visualization:** Plotly

## How It Works
1. User inputs health/lifestyle metrics (sleep, activity, stress, diet quality, etc.)
2. Calibrated ensemble models predict risk probabilities for each condition
3. Results are visualized against a synthetic cohort to contextualize individual risk
4. Data Explorer page allows deeper analysis of feature correlations and risk distributions

## Run Locally
```bash
git clone https://github.com/<your-username>/vitalsense-health-risk-predictor.git
cd vitalsense-health-risk-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This tool uses a synthetic dataset for demonstration purposes and is not intended for real medical diagnosis.

## Author
Prasandh S — AI & Data Science Student
