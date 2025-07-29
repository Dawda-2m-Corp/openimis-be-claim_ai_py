"""
Django REST Framework serializers for OpenIMIS Claim AI module
"""

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import (
    ClaimAIModel,
    ClaimPrediction,
    ClaimAIFeedback,
    ClaimAITrainingData,
    ClaimAIModelPerformance,
)


class UserSerializer(serializers.ModelSerializer):
    """Serializer for User model"""

    class Meta:
        model = User
        fields = ["id", "username", "first_name", "last_name", "email"]
        read_only_fields = ["id"]


class ClaimAIModelSerializer(serializers.ModelSerializer):
    """Serializer for AI models"""

    created_by = UserSerializer(read_only=True)
    updated_by = UserSerializer(read_only=True)
    prediction_count = serializers.SerializerMethodField()
    avg_confidence = serializers.SerializerMethodField()

    class Meta:
        model = ClaimAIModel
        fields = [
            "id",
            "name",
            "model_type",
            "version",
            "status",
            "description",
            "algorithm",
            "hyperparameters",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_score",
            "model_file",
            "scaler_file",
            "encoder_file",
            "created_at",
            "updated_at",
            "trained_at",
            "created_by",
            "updated_by",
            "prediction_count",
            "avg_confidence",
        ]
        read_only_fields = [
            "id",
            "created_at",
            "updated_at",
            "prediction_count",
            "avg_confidence",
        ]

    def get_prediction_count(self, obj):
        """Get prediction count for this model"""
        return obj.predictions.count()

    def get_avg_confidence(self, obj):
        """Get average confidence score for this model"""
        predictions = obj.predictions.all()
        if predictions:
            return sum(p.confidence_score for p in predictions) / len(predictions)
        return 0.0


class ClaimPredictionSerializer(serializers.ModelSerializer):
    """Serializer for claim predictions"""

    ai_model = ClaimAIModelSerializer(read_only=True)
    created_by = UserSerializer(read_only=True)
    feedback_count = serializers.SerializerMethodField()

    class Meta:
        model = ClaimPrediction
        fields = [
            "id",
            "claim_id",
            "prediction_type",
            "prediction_value",
            "confidence_score",
            "confidence_level",
            "ai_model",
            "input_features",
            "prediction_probabilities",
            "explanation",
            "feature_importance",
            "created_at",
            "updated_at",
            "created_by",
            "feedback_count",
        ]
        read_only_fields = ["id", "created_at", "updated_at", "feedback_count"]

    def get_feedback_count(self, obj):
        """Get feedback count for this prediction"""
        return obj.feedback.count()


class ClaimAIFeedbackSerializer(serializers.ModelSerializer):
    """Serializer for AI feedback"""

    prediction = ClaimPredictionSerializer(read_only=True)
    created_by = UserSerializer(read_only=True)

    class Meta:
        model = ClaimAIFeedback
        fields = [
            "id",
            "prediction",
            "feedback_type",
            "feedback_notes",
            "actual_outcome",
            "accuracy_rating",
            "usefulness_rating",
            "created_at",
            "updated_at",
            "created_by",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ClaimAITrainingDataSerializer(serializers.ModelSerializer):
    """Serializer for training data"""

    created_by = UserSerializer(read_only=True)

    class Meta:
        model = ClaimAITrainingData
        fields = [
            "id",
            "name",
            "description",
            "data_type",
            "data_file",
            "data_size",
            "feature_columns",
            "target_columns",
            "source",
            "version",
            "created_at",
            "updated_at",
            "created_by",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]


class ClaimAIModelPerformanceSerializer(serializers.ModelSerializer):
    """Serializer for model performance"""

    ai_model = ClaimAIModelSerializer(read_only=True)

    class Meta:
        model = ClaimAIModelPerformance
        fields = [
            "id",
            "ai_model",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "auc_score",
            "false_positive_rate",
            "false_negative_rate",
            "true_positive_rate",
            "true_negative_rate",
            "evaluation_date",
            "test_data_size",
            "evaluation_notes",
            "created_at",
        ]
        read_only_fields = ["id", "created_at"]


class PredictionRequestSerializer(serializers.Serializer):
    """Serializer for prediction requests"""

    claim_id = serializers.CharField(max_length=100)
    claim_data = serializers.DictField()
    model_id = serializers.UUIDField(required=False, allow_null=True)
    prediction_type = serializers.ChoiceField(
        choices=ClaimPrediction.PREDICTION_TYPES, default="approval"
    )


class TrainingRequestSerializer(serializers.Serializer):
    """Serializer for training requests"""

    name = serializers.CharField(max_length=255)
    model_type = serializers.ChoiceField(choices=ClaimAIModel.MODEL_TYPES)
    training_data_id = serializers.UUIDField()
    version = serializers.CharField(max_length=50, required=False, default="1.0.0")
    description = serializers.CharField(required=False, allow_blank=True)
    algorithm = serializers.CharField(
        max_length=100, required=False, default="RandomForest"
    )
    hyperparameters = serializers.DictField(required=False, default=dict)


class FeedbackRequestSerializer(serializers.Serializer):
    """Serializer for feedback requests"""

    prediction_id = serializers.UUIDField()
    feedback_type = serializers.ChoiceField(choices=ClaimAIFeedback.FEEDBACK_TYPES)
    feedback_notes = serializers.CharField(required=False, allow_blank=True)
    actual_outcome = serializers.CharField(
        max_length=100, required=False, allow_blank=True
    )
    accuracy_rating = serializers.IntegerField(
        min_value=1, max_value=5, required=False, allow_null=True
    )
    usefulness_rating = serializers.IntegerField(
        min_value=1, max_value=5, required=False, allow_null=True
    )


class ModelPerformanceRequestSerializer(serializers.Serializer):
    """Serializer for model performance evaluation"""

    ai_model_id = serializers.UUIDField()
    accuracy = serializers.FloatField(min_value=0.0, max_value=1.0)
    precision = serializers.FloatField(min_value=0.0, max_value=1.0)
    recall = serializers.FloatField(min_value=0.0, max_value=1.0)
    f1_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    auc_score = serializers.FloatField(min_value=0.0, max_value=1.0)
    false_positive_rate = serializers.FloatField(min_value=0.0, max_value=1.0)
    false_negative_rate = serializers.FloatField(min_value=0.0, max_value=1.0)
    true_positive_rate = serializers.FloatField(min_value=0.0, max_value=1.0)
    true_negative_rate = serializers.FloatField(min_value=0.0, max_value=1.0)
    test_data_size = serializers.IntegerField(min_value=1)
    evaluation_notes = serializers.CharField(required=False, allow_blank=True)
    evaluation_date = serializers.DateTimeField(required=False)


class DashboardStatsSerializer(serializers.Serializer):
    """Serializer for dashboard statistics"""

    models = serializers.DictField()
    predictions = serializers.DictField()
    feedback = serializers.DictField()
