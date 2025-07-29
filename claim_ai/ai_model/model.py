"""
AI Model for OpenIMIS Claim Processing
Uses scikit-learn algorithms for claim validation, fraud detection, and risk assessment.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import logging

logger = logging.getLogger(__name__)


class ClaimAIModel:
    """
    AI model for claim processing using multiple scikit-learn algorithms.

    Algorithms used:
    1. Random Forest - For claim approval/rejection classification
    2. Isolation Forest - For anomaly detection (fraud detection)
    3. One-Class SVM - For outlier detection
    4. Logistic Regression - For risk scoring
    """

    def __init__(self):
        self.random_forest = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )

        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expected 10% fraud
            random_state=42,
            n_estimators=100,
        )

        self.one_class_svm = OneClassSVM(
            kernel="rbf",
            nu=0.1,  # Expected fraction of outliers
            gamma="scale",
        )

        self.logistic_regression = LogisticRegression(
            random_state=42, max_iter=1000, class_weight="balanced"
        )

        self.scaler = StandardScaler()
        self.label_encoders = {}  # Dictionary to store label encoders for each categorical feature
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

        self.models = {
            "fraud_detection": self.isolation_forest,
            "claim_classification": self.random_forest,
            "risk_scoring": self.logistic_regression,
            "anomaly_detection": self.one_class_svm,
        }

        self.is_trained = False

    def _get_label_encoder(self, column_name):
        """
        Get or create a label encoder for a specific column.

        Args:
            column_name (str): Name of the categorical column

        Returns:
            LabelEncoder: Label encoder for the column
        """
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
        return self.label_encoders[column_name]

    def preprocess_claim_data(self, data, for_training=True):
        """
        Preprocess claim data for AI model training or prediction.

        Args:
            data (pd.DataFrame): Raw claim data
            for_training (bool): Whether this is for training (includes target variables)

        Returns:
            tuple: (X_features, X_text, y_approved, y_fraud, y_risk) or (X_features, X_text) for prediction
        """
        # Handle missing values
        data = data.fillna(0)

        # Extract numerical features
        numerical_features = [
            "claim_amount",
            "patient_age",
            "days_in_hospital",
            "number_of_procedures",
            "total_cost",
            "deductible_amount",
            "co_payment",
            "out_of_pocket_max",
        ]

        # Extract categorical features
        categorical_features = [
            "provider_type",
            "diagnosis_code",
            "procedure_code",
            "claim_type",
            "patient_gender",
            "insurance_type",
        ]

        # Extract text features
        text_features = ["claim_description", "diagnosis_notes"]

        X_numerical = data[numerical_features].values
        # X_categorical = data[categorical_features].values

        # Encode categorical variables
        if for_training:
            # For training, fit and transform
            X_categorical_encoded = np.column_stack(
                [
                    self._get_label_encoder(col).fit_transform(data[col])
                    for col in categorical_features
                ]
            )
        else:
            # For prediction, only transform (assuming already fitted)
            X_categorical_encoded = np.column_stack(
                [
                    self._get_label_encoder(col).transform(data[col])
                    for col in categorical_features
                ]
            )

        # Combine numerical and categorical features
        X_features = np.hstack([X_numerical, X_categorical_encoded])

        # Process text features
        text_data = data[text_features].fillna("").agg(" ".join, axis=1)
        if for_training:
            # For training, fit and transform
            X_text = self.tfidf_vectorizer.fit_transform(text_data)
        else:
            # For prediction, only transform
            X_text = self.tfidf_vectorizer.transform(text_data)

        if for_training:
            # Create target variables (only for training)
            y_approved = (data["claim_status"] == "approved").astype(int)
            y_fraud = (data["fraud_indicator"] == 1).astype(int)
            y_risk = data["risk_score"].values
            return X_features, X_text, y_approved, y_fraud, y_risk
        else:
            # For prediction, return only features
            return X_features, X_text

    def train_models(self, training_data):
        """
        Train all AI models on claim data.

        Args:
            training_data (pd.DataFrame): Training dataset
        """
        logger.info("Starting model training...")

        # Preprocess data
        X_features, X_text, y_approved, y_fraud, y_risk = self.preprocess_claim_data(
            training_data, for_training=True
        )

        # Combine features
        X_combined = np.hstack([X_features, X_text.toarray()])

        # Scale features
        X_scaled = self.scaler.fit_transform(X_combined)

        # Train Random Forest for claim classification
        logger.info("Training Random Forest for claim classification...")
        self.random_forest.fit(X_scaled, y_approved)

        # Train Isolation Forest for fraud detection
        logger.info("Training Isolation Forest for fraud detection...")
        self.isolation_forest.fit(X_scaled)

        # Train One-Class SVM for anomaly detection
        logger.info("Training One-Class SVM for anomaly detection...")
        # Use only non-fraudulent claims for training
        non_fraud_mask = y_fraud == 0
        if np.sum(non_fraud_mask) > 0:
            self.one_class_svm.fit(X_scaled[non_fraud_mask])

        # Train Logistic Regression for risk scoring
        logger.info("Training Logistic Regression for risk scoring...")
        self.logistic_regression.fit(X_scaled, y_risk)

        self.is_trained = True
        logger.info("All models trained successfully!")

    def predict_claim(self, claim_data):
        """
        Make predictions on a single claim.

        Args:
            claim_data (dict): Single claim data

        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([claim_data])

        # Preprocess
        X_features, X_text = self.preprocess_claim_data(df, for_training=False)
        X_combined = np.hstack([X_features, X_text.toarray()])
        X_scaled = self.scaler.transform(X_combined)

        # Make predictions
        approval_prob = self.random_forest.predict_proba(X_scaled)[0][1]
        fraud_score = self.isolation_forest.decision_function(X_scaled)[0]
        anomaly_score = self.one_class_svm.decision_function(X_scaled)[0]
        risk_score = self.logistic_regression.predict_proba(X_scaled)[0][1]

        # Determine fraud probability (lower score = more likely fraud)
        fraud_prob = 1 - (fraud_score + 1) / 2  # Normalize to [0, 1]

        return {
            "approval_probability": float(approval_prob),
            "fraud_probability": float(fraud_prob),
            "anomaly_score": float(anomaly_score),
            "risk_score": float(risk_score),
            "recommended_action": self._get_recommended_action(
                approval_prob, fraud_prob, risk_score
            ),
        }

    def _get_recommended_action(self, approval_prob, fraud_prob, risk_score):
        """
        Determine recommended action based on model predictions.

        Args:
            approval_prob (float): Approval probability
            fraud_prob (float): Fraud probability
            risk_score (float): Risk score

        Returns:
            str: Recommended action
        """
        if fraud_prob > 0.7:
            return "REJECT_FRAUD"
        elif fraud_prob > 0.5:
            return "REVIEW_FRAUD"
        elif approval_prob < 0.3:
            return "REJECT_LOW_APPROVAL"
        elif risk_score > 0.8:
            return "REVIEW_HIGH_RISK"
        elif approval_prob > 0.7 and fraud_prob < 0.3:
            return "APPROVE"
        else:
            return "MANUAL_REVIEW"

    def evaluate_models(self, test_data):
        """
        Evaluate model performance on test data.

        Args:
            test_data (pd.DataFrame): Test dataset

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating models...")

        # Preprocess test data
        X_features, X_text, y_approved, y_fraud, y_risk = self.preprocess_claim_data(
            test_data, for_training=True
        )
        X_combined = np.hstack([X_features, X_text.toarray()])
        X_scaled = self.scaler.transform(X_combined)

        # Evaluate Random Forest
        y_pred_approved = self.random_forest.predict(X_scaled)
        approval_accuracy = (y_pred_approved == y_approved).mean()
        approval_auc = roc_auc_score(
            y_approved, self.random_forest.predict_proba(X_scaled)[:, 1]
        )

        # Evaluate Isolation Forest
        fraud_scores = self.isolation_forest.decision_function(X_scaled)
        fraud_auc = roc_auc_score(
            y_fraud, -fraud_scores
        )  # Negative because lower score = fraud

        # Evaluate Logistic Regression
        y_pred_risk = self.logistic_regression.predict(X_scaled)
        risk_accuracy = (y_pred_risk == y_risk).mean()

        return {
            "claim_classification": {
                "accuracy": float(approval_accuracy),
                "auc": float(approval_auc),
            },
            "fraud_detection": {"auc": float(fraud_auc)},
            "risk_scoring": {"accuracy": float(risk_accuracy)},
        }

    def save_models(self, filepath):
        """
        Save trained models to disk.

        Args:
            filepath (str): Path to save models
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")

        model_data = {
            "models": self.models,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "tfidf_vectorizer": self.tfidf_vectorizer,
            "is_trained": self.is_trained,
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath):
        """
        Load trained models from disk.

        Args:
            filepath (str): Path to load models from
        """
        model_data = joblib.load(filepath)

        self.models = model_data["models"]
        self.scaler = model_data["scaler"]
        self.label_encoders = model_data["label_encoders"]
        self.tfidf_vectorizer = model_data["tfidf_vectorizer"]
        self.is_trained = model_data["is_trained"]

        # Update model references
        self.random_forest = self.models["claim_classification"]
        self.isolation_forest = self.models["fraud_detection"]
        self.one_class_svm = self.models["anomaly_detection"]
        self.logistic_regression = self.models["risk_scoring"]

        logger.info(f"Models loaded from {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Sample data generation for testing
    np.random.seed(42)
    n_samples = 1000

    sample_data = pd.DataFrame(
        {
            "claim_amount": np.random.exponential(5000, n_samples),
            "patient_age": np.random.normal(45, 15, n_samples),
            "days_in_hospital": np.random.poisson(3, n_samples),
            "number_of_procedures": np.random.poisson(2, n_samples),
            "total_cost": np.random.exponential(8000, n_samples),
            "deductible_amount": np.random.uniform(0, 2000, n_samples),
            "co_payment": np.random.uniform(0, 500, n_samples),
            "out_of_pocket_max": np.random.uniform(2000, 8000, n_samples),
            "provider_type": np.random.choice(
                ["hospital", "clinic", "specialist"], n_samples
            ),
            "diagnosis_code": np.random.choice(
                ["ICD10_A", "ICD10_B", "ICD10_C"], n_samples
            ),
            "procedure_code": np.random.choice(["CPT_A", "CPT_B", "CPT_C"], n_samples),
            "claim_type": np.random.choice(
                ["inpatient", "outpatient", "emergency"], n_samples
            ),
            "patient_gender": np.random.choice(["M", "F"], n_samples),
            "insurance_type": np.random.choice(
                ["private", "public", "corporate"], n_samples
            ),
            "claim_description": ["Sample claim description"] * n_samples,
            "diagnosis_notes": ["Sample diagnosis notes"] * n_samples,
            "claim_status": np.random.choice(
                ["approved", "rejected"], n_samples, p=[0.8, 0.2]
            ),
            "fraud_indicator": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "risk_score": np.random.choice(
                [0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]
            ),
        }
    )

    # Initialize and train model
    model = ClaimAIModel()
    model.train_models(sample_data)

    # Test prediction
    test_claim = {
        "claim_amount": 5000,
        "patient_age": 45,
        "days_in_hospital": 3,
        "number_of_procedures": 2,
        "total_cost": 8000,
        "deductible_amount": 1000,
        "co_payment": 200,
        "out_of_pocket_max": 5000,
        "provider_type": "hospital",
        "diagnosis_code": "ICD10_A",
        "procedure_code": "CPT_A",
        "claim_type": "inpatient",
        "patient_gender": "M",
        "insurance_type": "private",
        "claim_description": "Sample claim for testing",
        "diagnosis_notes": "Sample diagnosis for testing",
    }

    prediction = model.predict_claim(test_claim)
    print("Claim Prediction Results:")
    for key, value in prediction.items():
        print(f"{key}: {value}")
