from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, random_state=42):
        self.model = LogisticRegression(
            class_weight='balanced',
            random_state=random_state,
            max_iter=1000
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)