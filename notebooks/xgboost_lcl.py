from xgboost import XGBClassifier

def train_xgb(X_train, y_train):
    """
    Trains an XGBoost classifier for multi-class classification.
    """

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        eval_metric='mlogloss',
        # use_label_encoder=False,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
