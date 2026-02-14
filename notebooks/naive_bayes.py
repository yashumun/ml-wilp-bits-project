from sklearn.naive_bayes import GaussianNB
import joblib

def train_nb(X_train, y_train):
    """
    Trains a Gaussian Naive Bayes classifier.
    """
    model = GaussianNB()

    model.fit(X_train, y_train)
    return model
