# EMPLOYEE_ATTRITION_-Prediction-ML
An end-to-end Machine Learning pipeline utilizing SMOTE and advanced classifiers to predict corporate employee attrition using the IBM HR dataset
# Employee Attrition Prediction using Machine Learning

## 📌 Project Overview
This project builds an end-to-end predictive machine learning pipeline to identify high-risk attrition factors and proactively mitigate voluntary employee turnover. Using the classic IBM HR Analytics dataset, the project handles severe class imbalances using SMOTE and evaluates multiple classification models to deliver data-driven, actionable retention strategies for HR departments.

## 🛠️ Tech Stack & Frameworks Used
* **Languages:** Python
* **Data Processing & EDA:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning Pipeline:** Imbalanced-Learn (SMOTE), Scikit-Learn, XGBoost

## ⚙️ Project Pipeline & Architecture
1. **Data Audit & Hygiene:** Removed structurally irrelevant/zero-variance features (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).
2. **Exploratory Data Analysis (EDA):** Generated a matrix of 10+ professional visualizations identifying major operational turnover triggers (overtime, specific lower-paying job roles, and long commute distances).
3. **Ethical Machine Learning Splitting:** Split the dataset into 80/20 using stratified sampling, scaled numerical features, and applied **SMOTE** exclusively to the training partition to protect against data leakage.
4. **Model Evaluation:** Trained and cross-evaluated Logistic Regression, Random Forest Classifier, Support Vector Machine (SVM), and XGBoost Classifier with a heavy emphasis on maximizing **Recall**..
