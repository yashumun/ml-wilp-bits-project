
from sklearn.neighbors import KNeighborsClassifier
import joblib

def train_knn(X_train, y_train):
    """
    Trains a KNN classifier using already preprocessed data.
    """

    model = KNeighborsClassifier(
        n_neighbors=7,      # important: better than default 5 for this dataset
        metric="minkowski",
        p=2                 # Euclidean distance
    )

    model.fit(X_train, y_train)
    return model
