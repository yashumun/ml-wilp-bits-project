from notebooks.decision_tree.decision_tree import train_dt
from notebooks.helper.commoncode import create_data, get_metrics
from notebooks.knn.knn import train_knn
from notebooks.logistics_regression.logistics_regression import train_logistic
from notebooks.naive_bayes.naive_bayes import train_nb
from notebooks.random_forest.random_forest import train_rf
from notebooks.xgboost.xgboost import train_xgb
import pandas as pd
import os,joblib

def train_model():
    # prepare dataset (split + scaling from commoncode)
    df = pd.read_csv('../data/train.csv')
    X_train, X_test, y_train, y_test, scaler, feature_columns = create_data(df)
    logistics_model=train_logistic(X_train, y_train, scaler, feature_columns)
    get_metrics(y, y_pred, y_prob)
    decision_tree=train_dt(X_train, y_train)
    random_forest=train_rf(X_train, y_train)
    xgboost=train_xgb(X_train, y_train)
    knn=train_knn(X_train, y_train)
    naive_bayes=train_nb(X_train, y_train)


    # current file location: project/model/export_model.py
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    SAVE_DIR = os.path.join(PROJECT_ROOT, "model")
    # create folder automatically if missing
    os.makedirs(SAVE_DIR, exist_ok=True)

    # save files
    joblib.dump(logistics_model, os.path.join(SAVE_DIR, "logistics.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))
    joblib.dump(feature_columns, os.path.join(SAVE_DIR, "columns.pkl"))
    joblib.dump(decision_tree, os.path.join(SAVE_DIR, "decision_tree.pkl"))
    joblib.dump(random_forest, os.path.join(SAVE_DIR, "random_forest.pkl"))
    joblib.dump(naive_bayes, os.path.join(SAVE_DIR, "naive_bayes.pkl"))
    joblib.dump(knn, os.path.join(SAVE_DIR, "knn.pkl"))
    joblib.dump(xgboost, os.path.join(SAVE_DIR, "xgboost.pkl"))

train_model()
