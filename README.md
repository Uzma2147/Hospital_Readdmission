# Hospital_Readdmission
Predictive healthcare analytics pipeline analyzing patient clinical records and healthcare utilization to mitigate 30-day readmission risks
# 🏥 Hospital Readmission Prediction - End-to-End Machine Learning Capstone

An end-to-end classification pipeline designed to predict 30-day hospital readmissions for diabetic and general medical patients. Developed as **Task 4 (Capstone Project)** for the **INOVEGEN AI/ML Internship Program**.

##  Project Overview
Preventable 30-day hospital readmissions place a heavy financial and operational burden on healthcare systems while impacting patient outcomes. This project builds a robust binary classification system to analyze patient clinical histories, laboratory procedures, and healthcare utilization patterns to identify high-risk patients prior to discharge.

* **Dataset Size:** 25,000 patient records across 17 clinical and demographic features.
* **Target Variable:** `readmitted` (Binary: `1` for readmitted within 30 days, `0` for no readmission).
* **Class Balance:** Balanced target distribution (~53% non-readmitted vs. ~47% readmitted), enabling reliable metric evaluation without synthetic resampling.

---

##  Exploratory Data Analysis & Key Insights
Through exploratory data analysis and a $2 \times 2$ multi-plot visualization grid, several strong clinical risk indicators were uncovered:
1. **Prior Inpatient Visits (`n_inpatient`):** Patients with prior inpatient admissions exhibit a dramatically higher likelihood of 30-day readmission.
2. **Length of Stay (`time_in_hospital`):** Extended hospital stays correlate with higher clinical complexity and post-discharge vulnerability.
3. **Primary Diagnosis (`diag_1`):** Circulatory, diabetes, and respiratory conditions represent the top diagnostic drivers for hospital returns.
4. **Medication Density (`n_medications`):** Patients managing intricate pharmaceutical regimens show higher readmission density ratios.


##  Methodology & Pipeline Workflow
1. **Data Ingestion & Profiling:** Inspected missing values ($0$ nulls) and duplicate rows ($0$ duplicates).
2. **Feature Engineering:** Created aggregate clinical metrics (such as `total_visits`) and mapped target strings to binary integers.
3. **Encoding & Scaling:** Applied one-hot encoding to categorical features and standardized numerical scales using `StandardScaler`.
4. **Train-Test Split:** Performed an $80/20$ stratified split (`random_state=42`) to preserve class proportions.
5. **Model Training & Comparison:** Trained a baseline **Logistic Regression** model and an ensemble **Random Forest Classifier**.


##  Model Performance Evaluation

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Logistic Regression)** | 60.80% | 62.69% | 41.09% | 0.4964 | 0.6445 |
| **Ensemble (Random Forest)** | **61.58%** | **61.98%** | **47.30%** | **0.5366** | **0.6533** |

### Key Findings & Winning Model Selection:
* **Random Forest** was selected as the winning model due to superior **Recall ($47.30\%$)**, **F1-Score ($0.5366$)**, and **ROC-AUC ($0.6533$)**. 
* In a clinical setting, maximizing Recall is critical to ensure high-risk patients are successfully identified and flagged for post-discharge intervention.


##  Business & Healthcare Impact
* **Proactive Care Coordination:** Deploying this model allows hospital case managers to flag high-risk patients before discharge and schedule mandatory 48-hour follow-up appointments.
* **Resource Optimization:** Reduces preventable readmission penalties and optimizes hospital bed capacity.

##  Tech Stack & Dependencies
* **Language:** Python 3.13
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
* **Environment:** Google Colab & Jupyter Notebooks


##  Repository Structure
```text
├── hospital_readmissions.csv                 # Raw clinical dataset
├── Task_4_End_to_End_Predictive_Analytics.ipynb # Complete executable notebook
└── README.md                                 # Project documentation
