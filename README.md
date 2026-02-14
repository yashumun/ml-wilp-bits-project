# 📱 Mobile Price Range Classification --- Machine Learning Web App

## 1. Problem Statement

The objective of this project is to build and evaluate multiple machine
learning classification models that can predict the **price category of
a mobile phone** based on its hardware specifications.

Given various features such as battery capacity, RAM, internal memory,
processor speed, and connectivity options, the model classifies a mobile
device into one of four categories:

-   0 → Low Cost
-   1 → Medium Cost
-   2 → High Cost
-   3 → Very High Cost

An interactive **Streamlit web application** has also been developed to
allow users to upload a dataset and validate the trained models.

------------------------------------------------------------------------

## 2. Dataset Description

Dataset used: **Mobile Price Classification Dataset (Kaggle)**

The dataset contains technical specifications of mobile phones and their
corresponding price range categories.

### Dataset Characteristics

-   Total Instances: 1800
-   Number of Features: 20
-   Target Variable: `price_range`
-   Type: Multi-class classification
-   Missing Values: None

### Important Features

Some key features used in prediction: - `battery_power` -- Battery
capacity (mAh) - `ram` -- RAM size - `px_height`, `px_width` -- Screen
resolution - `clock_speed` -- Processor speed - `int_memory` -- Internal
storage - `mobile_wt` -- Mobile weight - `wifi`, `bluetooth`, `dual_sim`
-- Connectivity features

------------------------------------------------------------------------

## 3. Machine Learning Models Used

The following classification algorithms were implemented and evaluated
on the same dataset:

1.  Logistic Regression
2.  Decision Tree Classifier
3.  K-Nearest Neighbors (KNN)
4.  Naive Bayes
5.  Random Forest
6.  XGBoost

All models were trained locally and exported as serialized `.pkl`
files. The Streamlit app loads these trained models for validation.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Each model is evaluated using the following performance metrics:

-   Accuracy
-   Precision
-   Recall 
-   F1 Score
-   AUC Score (One-vs-Rest for multi-class)
-   Matthews Correlation Coefficient (MCC)

A confusion matrix and classification report are also displayed in the
UI.

------------------------------------------------------------------------

## 5. Model Comparison Table

| ML Model  | Accuracy  | AUC  | Precision  |  Recall | F1 Score  | MCC  |
|---|---|---|---|---|---|---|
| Logistic Regression  |   |   |   |   |   |   |
| Decision Tree  |   |   |   |   |   |   |
|  KNN |   |   |   |   |   |   |
| Naive Bayes  |   |   |   |   |   |   |
|  Random Forest |   |   |   |   |   |   |
|  XGBoost |   |   |   |   |   |   |
                                                                   

------------------------------------------------------------------------

## 6. Observations

| **Model**               | **Observation**                                                                                                |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| **Logistic Regression** | Performs well as a baseline model with stable results due to linear separability of some features.             |
| **Decision Tree**       | Faster but tends to overfit the training data, leading to slightly lower generalization performance.           |
| **KNN**                 | Performs better after feature scaling but computationally slower for large datasets.                           |
| **Naive Bayes**         | Fastest model but lower accuracy because the independence assumption between features is unrealistic.          |
| **Random Forest**       | Provides strong performance due to ensemble averaging and reduces overfitting.                                 |
| **XGBoost**             | Best performing model overall as it captures complex nonlinear relationships and optimizes errors iteratively. |


------------------------------------------------------------------------

------------------------------------------------------------------------

## 7. Streamlit Web Application Features

The developed web application allows users to:

-   Select a trained model from the sidebar
-   Upload a validation dataset (CSV format)
-   Automatically preprocess data using saved scaler
-   Predict price categories
-   Display evaluation metrics
-   Visualize confusion matrix (heatmap)
-   View classification report

------------------------------------------------------------------------

## 8. Project Structure

    project/
    │
    ├── data/
    ├── app.py                 # Streamlit application
    ├── helper.py              # Metrics calculation
    ├── model/
    │   ├── *.pkl              # Trained models
    │   ├── scaler.pkl
    │   └── columns.pkl
    ├── notebooks/            # Training and experimentation
    │    └── export_models.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 9. How to Run the Project Locally

### Step 1: Install dependencies

``` bash
pip install -r requirements.txt
```

### Step 2: Run Streamlit app

``` bash
streamlit run app.py
```

The application will open automatically in your browser.

------------------------------------------------------------------------

## 10. Deployment

The application is deployed using **Streamlit Community Cloud**.

Live App Link: *https://ml-wilp-bits-project-yashmun.streamlit.app/*
GitHub Repository: *https://github.com/yashumun/ml-wilp-bits-project*

https://share.streamlit.io/user/yashumun

------------------------------------------------------------------------

## 11. Technologies Used

-   Python
-   Scikit-learn
-   XGBoost
-   Pandas & NumPy
-   Matplotlib & Seaborn
-   Streamlit

------------------------------------------------------------------------

## 12. Conclusion

This project demonstrates an end-to-end machine learning workflow
including: data preprocessing, model training, evaluation,
visualization, and deployment as an interactive web application.

Among all models, ensemble methods (Random Forest and XGBoost) achieved
the highest performance due to their ability to capture nonlinear
feature interactions and reduce overfitting.
