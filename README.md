# Mobile Price Range Classification --- Machine Learning Web App

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

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|------|------|------|------|------|------|------|
| Logistic Regression | 0.970 | 0.9988 | 0.9705 | 0.970 | 0.9700 | 0.9602 |
| Decision Tree | 0.830 | 0.8867 | 0.8297 | 0.830 | 0.8298 | 0.7734 |
| KNN | 0.525 | 0.7625 | 0.5631 | 0.525 | 0.5305 | 0.3724 |
| Naive Bayes | 0.830 | 0.9634 | 0.8358 | 0.830 | 0.8322 | 0.7738 |
| Random Forest | 0.940 | 0.9887 | 0.9423 | 0.940 | 0.9402 | 0.9206 |
| XGBoost | 0.940 | 0.9964 | 0.9409 | 0.940 | 0.9403 | 0.9201 |

------------------------------------------------------------------------

## 6. Observations

| Model | Observation |
|------|------|
| Logistic Regression | Highest performance; achieved the highest accuracy (97%) and the highest MCC (0.96), along with an almost perfect AUC score (0.9988). This indicates that after feature scaling, the dataset becomes close to linearly separable |
| Random Forest | Strong ensemble model capturing feature interactions and reducing overfitting. |
|  XGBoost  | Ensemble methods successfully captured complex feature interactions and produced stable predictions, but they did not surpass Logistic Regression, suggesting that excessive model complexity was not necessary for this dataset. |
|  Decision Tree | Performs reasonably but prone to overfitting compared to ensembles. |
|  Naive Bayes | Good performance but limited by independence assumption between features. |
|  KNN | Lowest performance due to curse of dimensionality in 20-feature space. |


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
    ├── model/
    │   ├── *.pkl              # Trained models
    │   ├── scaler.pkl
    │   └── columns.pkl
    ├── notebooks/            # Training and experimentation
    |    ├── helper/              # Metrics calculation
    |    |      |_commoncode.py
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


------------------------------------------------------------------------

## 11. Technologies Used

-   Python
-   Scikit-learn
-   XGBoost
-   Pandas & NumPy
-   Matplotlib & Seaborn
-   Streamlit

