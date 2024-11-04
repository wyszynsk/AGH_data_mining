import unittest
import numpy as np
from regresion import LogisticRegressor 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import time

class TestLogisticRegressor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        
        cls.X, cls.y = make_classification(
            n_samples=1000,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=42
        )
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.25, random_state=42)

    def test_initialization(self):
        """Test if the model initializes with the correct parameters."""
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        self.assertEqual(model.learning_rate, 0.05)
        self.assertEqual(model.iterations, 1000)

    def test_training(self):
        """Test if the model can train without errors and updates weights correctly."""
        model = LogisticRegressor(learning_rate=0.05, iterations=100)
        model.fit(self.X_train, self.y_train)
        
        # check if weights and bias are set after training
        self.assertIsNotNone(model.w, "Weights should not be None after training.")
        self.assertIsNotNone(model.b, "Bias should not be None after training.")
        # check that weights have non-zero values
        self.assertFalse(np.all((model.w == 0)), "Weights should be updated and not remain zero.")

    def test_prediction(self):
        """Test if the model can predict and output correct shape."""
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        
        # check if the prediction has the correct length
        self.assertEqual(len(y_pred), len(self.y_test), "Prediction output shape mismatch.")
        
    def test_accuracy(self):
        # test if the model accuracy is within a reasonable range compared to sklearns
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        # compare with sklearn's LogisticRegression 
        model_sk = LogisticRegression()
        model_sk.fit(self.X_train, self.y_train)
        y_pred_sk = model_sk.predict(self.X_test)
        accuracy_sk = accuracy_score(self.y_test, y_pred_sk)

        # assert that my model's accuracy is reasonably close to sklearn's
        self.assertAlmostEqual(accuracy, accuracy_sk, delta=0.1, msg="Custom model accuracy should be close to sklearn model accuracy.")
        
    def test_edge_cases(self):
        # test edge cases i.e.  when no data is provided or when all labels are the same
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        
        # empty dataset
        with self.assertRaises(ValueError, msg="Model should raise an error on empty dataset."):
            model.fit(np.array([]), np.array([]))
        
        # single class dataset (all zeros or all ones)
        y_single_class = np.zeros(len(self.y_train))
        with self.assertRaises(ValueError, msg="Model should raise an error on single-class dataset."):
            model.fit(self.X_train, y_single_class)


    # Timing tests
    def test_fit_timing(self):
       
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        repetitions = 1000  
        fit_times = []

        for _ in range(repetitions):
            start_time = time.time()
            model.fit(self.X_train, self.y_train)
            end_time = time.time()
            fit_times.append(end_time - start_time)

        avg_fit_time = np.mean(fit_times)
        print(f"Average fit method execution time over {repetitions} runs: {avg_fit_time} seconds")
        self.assertLess(avg_fit_time, 1.0, "Fit method took too long on average")


    def test_predict_timing(self):
      
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        model.fit(self.X_train, self.y_train)
        repetitions = 1000
        predict_times = []

        for _ in range(repetitions):
            start_time = time.time()
            model.predict(self.X_test)
            end_time = time.time()
            predict_times.append(end_time - start_time)

        avg_predict_time = np.mean(predict_times)
        print(f"Average predict method execution time over {repetitions} runs: {avg_predict_time} seconds")
        self.assertLess(avg_predict_time, 0.1, "Predict method took too long on average")

    def test_update_weights_timing(self):
     
        model = LogisticRegressor(learning_rate=0.05, iterations=1000)
        model.X = self.X_train
        model.Y = self.y_train
        model.m, model.n = model.X.shape
        model.w = np.zeros(model.n)
        model.b = 0
        repetitions = 1000
        update_times = []

        for _ in range(repetitions):
            start_time = time.time()
            model.update_weights()
            end_time = time.time()
            update_times.append(end_time - start_time)

        avg_update_time = np.mean(update_times)
        print(f"Average update weights method execution time over {repetitions} runs: {avg_update_time} seconds")
        self.assertLess(avg_update_time, 0.01, "Update weights method took too long on average")


if __name__ == '__main__':
    unittest.main()
