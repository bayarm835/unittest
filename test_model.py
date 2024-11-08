# test_model.py
import unittest
from model import RegressionModel
from dataset import load_data

class TestRegressionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Configurer les données et le modèle une fois pour tous les tests."""
        # Charger les données d'entraînement et de test
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = load_data()
        
        # Initialiser l'instance du modèle
        cls.model = RegressionModel()

    def test_training(self):
        """Tester que le modèle est correctement entraîné."""
        # Entraîner le modèle
        self.model.train(self.X_train, self.y_train)
        
        # Vérifier que le modèle est marqué comme entraîné
        self.assertTrue(self.model.is_trained)

    def test_prediction(self):
        """Tester que le modèle peut faire des prédictions après l'entraînement."""
        # Assurer que le modèle est entraîné
        self.model.train(self.X_train, self.y_train)
        
        # Faire des prédictions
        predictions = self.model.predict(self.X_test)
        
        # Vérifier la longueur des prédictions
        self.assertEqual(len(predictions), len(self.y_test))

    def test_evaluate_metrics(self):
        """Tester que les métriques de performance sont calculées correctement."""
        # Entraîner le modèle et faire des prédictions
        self.model.train(self.X_train, self.y_train)
        predictions = self.model.predict(self.X_test)
        
        # Calculer les métriques
        metrics = self.model.evaluate(self.y_test, predictions)
        
        # Vérifier les valeurs des métriques
        self.assertIn("mean_squared_error", metrics)
        self.assertIn("mean_absolute_error", metrics)
        self.assertIn("r2_score", metrics)

        # Vérifier que les métriques sont des valeurs numériques
        for metric in metrics.values():
            self.assertIsInstance(metric, float)

# Exécution des tests unitaires
if __name__ == "__main__":
    unittest.main()
