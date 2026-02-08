from sklearn.ensemble import RandomForestClassifier
import joblib

def train_rf(X_train, y_train):
    """
    Trains a Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,   # stabilizes results
        random_state=42,
        n_jobs=-1           # uses all CPU cores (faster training)
    )

    model.fit(X_train, y_train)
    return model
