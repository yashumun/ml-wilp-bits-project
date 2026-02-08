from sklearn.tree import DecisionTreeClassifier
import joblib

def train_dt(X_train, y_train):
    """
    Trains a Decision Tree classifier using already preprocessed data.

    Parameters:
        X_train : training features
        y_train : training labels

    Returns:
        model : trained DecisionTreeClassifier
    """

    model = DecisionTreeClassifier(
        criterion="gini",     # default and works well for this dataset
        max_depth=None,       # allow tree to grow fully
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
