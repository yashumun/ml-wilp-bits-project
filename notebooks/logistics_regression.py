from sklearn.linear_model import LogisticRegression


def train_logistic(X_train, y_train, scaler, feature_columns):
    """
    Trains a Logistic Regression classifier using already preprocessed data.

    Parameters:
        X_train : training feature matrix (already scaled)
        y_train : training labels

    Returns:
        model : trained LogisticRegression model
    """

    model = LogisticRegression(
        max_iter=5000,          # allows convergence
        # multi_class='auto',     # handles multiclass automatically
        solver='lbfgs',         # stable solver for multiclass
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


