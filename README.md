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

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|--------|-----|------|------|------|------|
| Logistic Regression | 0.840 | 0.9592 | 0.8410 | 0.840 | 0.8401 | 0.7839 |
| Decision Tree | 0.796 | 0.8650 | 0.8007 | 0.796 | 0.7969 | 0.7244 |
| KNN | 0.624 | 0.8440 | 0.6413 | 0.624 | 0.6285 | 0.4897 |
| Naive Bayes | 0.416 | 0.7551 | 0.4967 | 0.416 | 0.3743 | 0.3123 |
| Random Forest | 0.812 | 0.9702 | 0.8168 | 0.812 | 0.8120 | 0.7475 |
| XGBoost | 0.836 | 0.9795 | 0.8387 | 0.836 | 0.8357 | 0.7795 |

------------------------------------------------------------------------

## 6. Observations
| Model | Observation |
|------|------|
| Logistic Regression | Strong baseline performance and stable results across all classes. |
| Decision Tree | Performs reasonably well but slightly prone to overfitting compared to ensembles. |
| KNN | Lower accuracy due to sensitivity to feature distance in higher-dimensional space. |
| Naive Bayes | Lowest performance because the feature independence assumption does not hold for this dataset. |
| Random Forest | High accuracy and robust predictions due to ensemble averaging. |
| XGBoost | Best overall model with highest AUC, showing strong capability to capture complex feature relationships. |


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
├── app.py                 # Streamlit web application
├── README.md              # Project documentation
├── requirements.txt       # Dependencies
├── .gitignore
│
├── data/
│   ├── train.csv          # Training dataset
│   └── links.txt
│
├── model/                 # Serialized trained models
│   ├── logistic.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── columns.pkl
│
└── notebooks/             # Model development & training
    ├── jupyter_all.ipynb
    ├── export_models.py
    │
    ├── helper/
    │   └── commoncode.py
    │
    ├── decision_tree/
    │   └── decision_tree.py
    │
    ├── logistics_regression/
    │   └── logistics_regression.py
    │
    ├── knn/
    │   └── knn.py
    │
    ├── naive_bayes/
    │   └── naive_bayes.py
    │
    ├── random_forest/
    │   └── random_forest.py
    │
    └── xgboost/
        └── xgboost.py


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

------------------------------------------------------------------------

## 12. Conclusion

This project demonstrates an end-to-end machine learning workflow
including: data preprocessing, model training, evaluation,
visualization, and deployment as an interactive web application.

Among all models, ensemble methods (Random Forest and XGBoost) achieved
the highest performance due to their ability to capture nonlinear
feature interactions and reduce overfitting.
