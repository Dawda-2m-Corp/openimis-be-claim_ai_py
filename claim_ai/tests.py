"""
Test cases for OpenIMIS Claim AI module
"""

import json
import uuid
from datetime import datetime, timedelta
from django.test import TestCase, Client
from django.contrib.auth.models import User
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from unittest.mock import patch, MagicMock

from .models import (
    ClaimAIModel,
    ClaimPrediction,
    ClaimAIFeedback,
    ClaimAITrainingData,
    ClaimAIModelPerformance,
)


class ClaimAIModelTestCase(TestCase):
    """Test cases for ClaimAIModel"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Fraud Detection Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            description="Test model for fraud detection",
            algorithm="RandomForest",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_score=0.87,
            created_by=self.user,
        )

    def test_model_creation(self):
        """Test AI model creation"""
        self.assertEqual(self.ai_model.name, "Test Fraud Detection Model")
        self.assertEqual(self.ai_model.model_type, "fraud_detection")
        self.assertEqual(self.ai_model.status, "active")
        self.assertEqual(self.ai_model.accuracy, 0.85)

    def test_model_str_representation(self):
        """Test string representation"""
        expected = "Test Fraud Detection Model v1.0.0 (Fraud Detection)"
        self.assertEqual(str(self.ai_model), expected)

    def test_model_validation(self):
        """Test model validation"""
        # Test valid model
        self.assertTrue(self.ai_model.pk)

        # Test invalid model type
        with self.assertRaises(Exception):
            ClaimAIModel.objects.create(
                name="Invalid Model",
                model_type="invalid_type",
                version="1.0.0",
                created_by=self.user,
            )


class ClaimPredictionTestCase(TestCase):
    """Test cases for ClaimPrediction"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

        self.prediction = ClaimPrediction.objects.create(
            claim_id="CLAIM-001",
            prediction_type="fraud",
            prediction_value=0.75,
            confidence_score=0.85,
            confidence_level="high",
            ai_model=self.ai_model,
            input_features={"amount": 5000, "age": 45},
            prediction_probabilities={"fraud": 0.75, "legitimate": 0.25},
            explanation="High risk due to unusual patterns",
            feature_importance={"amount": 0.6, "age": 0.4},
            created_by=self.user,
        )

    def test_prediction_creation(self):
        """Test prediction creation"""
        self.assertEqual(self.prediction.claim_id, "CLAIM-001")
        self.assertEqual(self.prediction.prediction_type, "fraud")
        self.assertEqual(self.prediction.prediction_value, 0.75)
        self.assertEqual(self.prediction.confidence_level, "high")

    def test_prediction_str_representation(self):
        """Test string representation"""
        expected = f"Prediction {self.prediction.id} for claim CLAIM-001"
        self.assertEqual(str(self.prediction), expected)

    def test_confidence_level_validation(self):
        """Test confidence level validation"""
        # Test valid confidence score
        self.prediction.confidence_score = 0.9
        self.prediction.save()
        self.assertEqual(self.prediction.confidence_level, "high")

        # Test medium confidence
        self.prediction.confidence_score = 0.7
        self.prediction.save()
        self.assertEqual(self.prediction.confidence_level, "medium")

        # Test low confidence
        self.prediction.confidence_score = 0.5
        self.prediction.save()
        self.assertEqual(self.prediction.confidence_level, "low")


class ClaimAIFeedbackTestCase(TestCase):
    """Test cases for ClaimAIFeedback"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

        self.prediction = ClaimPrediction.objects.create(
            claim_id="CLAIM-001",
            prediction_type="fraud",
            prediction_value=0.75,
            confidence_score=0.85,
            confidence_level="high",
            ai_model=self.ai_model,
            created_by=self.user,
        )

        self.feedback = ClaimAIFeedback.objects.create(
            prediction=self.prediction,
            feedback_type="correct",
            feedback_notes="Prediction was accurate",
            actual_outcome="fraud_detected",
            accuracy_rating=5,
            usefulness_rating=4,
            created_by=self.user,
        )

    def test_feedback_creation(self):
        """Test feedback creation"""
        self.assertEqual(self.feedback.feedback_type, "correct")
        self.assertEqual(self.feedback.accuracy_rating, 5)
        self.assertEqual(self.feedback.usefulness_rating, 4)

    def test_feedback_str_representation(self):
        """Test string representation"""
        expected = f"Feedback for prediction {self.prediction.id}"
        self.assertEqual(str(self.feedback), expected)


class ClaimAITrainingDataTestCase(TestCase):
    """Test cases for ClaimAITrainingData"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        # Create a test file
        self.test_file = SimpleUploadedFile(
            "test_data.csv",
            b"claim_id,amount,age,fraud\nCLAIM-001,5000,45,1\nCLAIM-002,3000,30,0",
            content_type="text/csv",
        )

        self.training_data = ClaimAITrainingData.objects.create(
            name="Test Training Data",
            description="Test dataset for fraud detection",
            data_type="training",
            data_file=self.test_file,
            data_size=1000,
            feature_columns=["claim_id", "amount", "age"],
            target_columns=["fraud"],
            source="test_source",
            version="1.0.0",
            created_by=self.user,
        )

    def test_training_data_creation(self):
        """Test training data creation"""
        self.assertEqual(self.training_data.name, "Test Training Data")
        self.assertEqual(self.training_data.data_type, "training")
        self.assertEqual(self.training_data.data_size, 1000)

    def test_training_data_str_representation(self):
        """Test string representation"""
        expected = "Test Training Data (Training Data)"
        self.assertEqual(str(self.training_data), expected)


class ClaimAIModelPerformanceTestCase(TestCase):
    """Test cases for ClaimAIModelPerformance"""

    def setUp(self):
        """Set up test data"""
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

        self.performance = ClaimAIModelPerformance.objects.create(
            ai_model=self.ai_model,
            accuracy=0.85,
            precision=0.82,
            recall=0.88,
            f1_score=0.85,
            auc_score=0.87,
            false_positive_rate=0.15,
            false_negative_rate=0.12,
            true_positive_rate=0.88,
            true_negative_rate=0.85,
            evaluation_date=datetime.now(),
            test_data_size=1000,
            evaluation_notes="Test evaluation",
        )

    def test_performance_creation(self):
        """Test performance creation"""
        self.assertEqual(self.performance.accuracy, 0.85)
        self.assertEqual(self.performance.precision, 0.82)
        self.assertEqual(self.performance.recall, 0.88)

    def test_performance_str_representation(self):
        """Test string representation"""
        expected = (
            f"Performance for Test Model on {self.performance.evaluation_date.date()}"
        )
        self.assertEqual(str(self.performance), expected)


class APIViewsTestCase(APITestCase):
    """Test cases for API views"""

    def setUp(self):
        """Set up test data"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.client.force_authenticate(user=self.user)

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

    @patch("claim_ai.views.AIModelClass")
    def test_predict_claim(self, mock_ai_model):
        """Test claim prediction API"""
        # Mock the AI model
        mock_instance = MagicMock()
        mock_instance.predict_claim.return_value = {
            "approval_probability": 0.75,
            "confidence_score": 0.85,
            "probabilities": {"approved": 0.75, "rejected": 0.25},
            "explanation": "Test explanation",
            "feature_importance": {"amount": 0.6},
        }
        mock_ai_model.return_value = mock_instance

        # Test data
        data = {
            "claim_id": "CLAIM-001",
            "claim_data": {"amount": 5000, "age": 45, "provider_type": "hospital"},
            "prediction_type": "fraud",
        }

        response = self.client.post("/api/v1/predict/", data, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("prediction", response.data)
        self.assertIn("ai_model", response.data)

    def test_predict_claim_missing_fields(self):
        """Test prediction with missing fields"""
        data = {"claim_id": "CLAIM-001"}  # Missing claim_data

        response = self.client.post("/api/v1/predict/", data, format="json")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("error", response.data)

    def test_provide_feedback(self):
        """Test feedback API"""
        # Create a prediction first
        prediction = ClaimPrediction.objects.create(
            claim_id="CLAIM-001",
            prediction_type="fraud",
            prediction_value=0.75,
            confidence_score=0.85,
            confidence_level="high",
            ai_model=self.ai_model,
            created_by=self.user,
        )

        # Test feedback data
        data = {
            "prediction_id": str(prediction.id),
            "feedback_type": "correct",
            "feedback_notes": "Prediction was accurate",
            "actual_outcome": "fraud_detected",
            "accuracy_rating": 5,
            "usefulness_rating": 4,
        }

        response = self.client.post("/api/v1/feedback/", data, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("feedback", response.data)

    def test_dashboard_stats(self):
        """Test dashboard statistics API"""
        # Create some test data
        ClaimPrediction.objects.create(
            claim_id="CLAIM-001",
            prediction_type="fraud",
            prediction_value=0.75,
            confidence_score=0.85,
            confidence_level="high",
            ai_model=self.ai_model,
            created_by=self.user,
        )

        response = self.client.get("/api/v1/dashboard/stats/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("models", response.data)
        self.assertIn("predictions", response.data)
        self.assertIn("feedback", response.data)


class ModelViewSetTestCase(APITestCase):
    """Test cases for ModelViewSet"""

    def setUp(self):
        """Set up test data"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.client.force_authenticate(user=self.user)

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

    def test_list_models(self):
        """Test listing AI models"""
        response = self.client.get("/api/v1/models/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_models_by_type(self):
        """Test filtering models by type"""
        response = self.client.get("/api/v1/models/?model_type=fraud_detection")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_models_by_status(self):
        """Test filtering models by status"""
        response = self.client.get("/api/v1/models/?status=active")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_search_models(self):
        """Test searching models"""
        response = self.client.get("/api/v1/models/?search=Test")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)


class PredictionViewSetTestCase(APITestCase):
    """Test cases for PredictionViewSet"""

    def setUp(self):
        """Set up test data"""
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )
        self.client.force_authenticate(user=self.user)

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

        self.prediction = ClaimPrediction.objects.create(
            claim_id="CLAIM-001",
            prediction_type="fraud",
            prediction_value=0.75,
            confidence_score=0.85,
            confidence_level="high",
            ai_model=self.ai_model,
            created_by=self.user,
        )

    def test_list_predictions(self):
        """Test listing predictions"""
        response = self.client.get("/api/v1/predictions/")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_predictions_by_claim_id(self):
        """Test filtering predictions by claim ID"""
        response = self.client.get("/api/v1/predictions/?claim_id=CLAIM-001")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_predictions_by_type(self):
        """Test filtering predictions by type"""
        response = self.client.get("/api/v1/predictions/?prediction_type=fraud")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data["results"]), 1)


class IntegrationTestCase(TestCase):
    """Integration test cases"""

    def setUp(self):
        """Set up test data"""
        self.client = Client()
        self.user = User.objects.create_user(
            username="testuser", email="test@example.com", password="testpass123"
        )

        self.ai_model = ClaimAIModel.objects.create(
            name="Test Model",
            model_type="fraud_detection",
            version="1.0.0",
            status="active",
            created_by=self.user,
        )

    def test_model_dashboard_view(self):
        """Test model dashboard view"""
        self.client.force_login(self.user)
        response = self.client.get("/dashboard/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("models", response.context)
        self.assertIn("recent_predictions", response.context)

    def test_legacy_predict_endpoint(self):
        """Test legacy prediction endpoint"""
        data = {"claim_id": "CLAIM-001", "claim_data": {"amount": 5000}}

        response = self.client.post(
            "/legacy/predict/", json.dumps(data), content_type="application/json"
        )

        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertEqual(response_data["status"], "success")
