from xgboost import XGBClassifier

class XGBoostModel:
    def __init__(self, n_estimators=200, max_depth=6, random_state=42):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            scale_pos_weight=5, 
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)