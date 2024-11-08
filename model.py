# model.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionModel:
    def __init__(self):
        # Initialiser le modèle de régression linéaire
        self.model = LinearRegression()
        self.is_trained = False  # Variable pour vérifier si le modèle est entraîné

    def train(self, X_train, y_train):
        """Entraîner le modèle sur les données d'entraînement."""
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, X_test):
        """Effectuer des prédictions sur les données de test."""
        if not self.is_trained:
            raise ValueError("Le modèle n'est pas entraîné.")
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        """Évaluer les performances du modèle avec différentes métriques de régression."""
        return {
            "mean_squared_error": mean_squared_error(y_true, y_pred),
            "mean_absolute_error": mean_absolute_error(y_true, y_pred),
            "r2_score": r2_score(y_true, y_pred),
        }
