
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

# def load_data():
#     df = pd.read_csv("../data/train.csv")
#     df.shape
#     return df

def create_data(df):


    X = df.drop("price_range", axis=1)
    y = df["price_range"]
    feature_columns = X.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.1,
        random_state=42,
        stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, feature_columns

def prediction(model,xtest):
    y_pred = model.predict(xtest)
    y_prob = model.predict_proba(xtest)
    return y_pred,y_prob

def get_metrics(y_test, y_pred, y_prob=None):

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted'),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    if y_prob is not None:
        metrics["AUC"] = roc_auc_score(y_test, y_prob, multi_class='ovr')
    else:
        metrics["AUC"] = None

    return metrics