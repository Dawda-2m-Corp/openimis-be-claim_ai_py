"""
Django models for OpenIMIS Claim AI module
"""

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.translation import gettext_lazy as _
import uuid


class ClaimAIModel(models.Model):
    """
    AI Model configuration and metadata
    """

    MODEL_TYPES = [
        ("fraud_detection", _("Fraud Detection")),
        ("claim_classification", _("Claim Classification")),
        ("risk_scoring", _("Risk Scoring")),
        ("anomaly_detection", _("Anomaly Detection")),
        ("quality_assessment", _("Quality Assessment")),
    ]

    STATUS_CHOICES = [
        ("active", _("Active")),
        ("inactive", _("Inactive")),
        ("training", _("Training")),
        ("testing", _("Testing")),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255, unique=True)
    model_type = models.CharField(max_length=50, choices=MODEL_TYPES)
    version = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="inactive")

    # Model metadata
    description = models.TextField(blank=True)
    algorithm = models.CharField(max_length=100)
    hyperparameters = models.JSONField(default=dict)

    # Performance metrics
    accuracy = models.FloatField(null=True, blank=True)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1_score = models.FloatField(null=True, blank=True)
    auc_score = models.FloatField(null=True, blank=True)

    # File storage
    model_file = models.FileField(upload_to="ai_models/", null=True, blank=True)
    scaler_file = models.FileField(upload_to="ai_models/", null=True, blank=True)
    encoder_file = models.FileField(upload_to="ai_models/", null=True, blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    trained_at = models.DateTimeField(null=True, blank=True)

    # Relationships
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="created_ai_models"
    )
    updated_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="updated_ai_models"
    )

    class Meta:
        db_table = "claim_ai_model"
        verbose_name = _("AI Model")
        verbose_name_plural = _("AI Models")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} v{self.version} ({self.get_model_type_display()})"


class ClaimPrediction(models.Model):
    """
    AI predictions for individual claims
    """

    PREDICTION_TYPES = [
        ("fraud", _("Fraud Detection")),
        ("approval", _("Approval Prediction")),
        ("risk", _("Risk Assessment")),
        ("quality", _("Quality Assessment")),
        ("anomaly", _("Anomaly Detection")),
    ]

    CONFIDENCE_LEVELS = [
        ("low", _("Low")),
        ("medium", _("Medium")),
        ("high", _("High")),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Claim reference (assuming integration with openIMIS claim system)
    claim_id = models.CharField(max_length=100, db_index=True)

    # Prediction details
    prediction_type = models.CharField(max_length=20, choices=PREDICTION_TYPES)
    prediction_value = models.FloatField()
    confidence_score = models.FloatField(
        validators=[MinValueValidator(0.0), MaxValueValidator(1.0)]
    )
    confidence_level = models.CharField(max_length=10, choices=CONFIDENCE_LEVELS)

    # Model used
    ai_model = models.ForeignKey(
        ClaimAIModel, on_delete=models.CASCADE, related_name="predictions"
    )

    # Features used for prediction
    input_features = models.JSONField(default=dict)
    prediction_probabilities = models.JSONField(default=dict)

    # Explanation and reasoning
    explanation = models.TextField(blank=True)
    feature_importance = models.JSONField(default=dict)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Relationships
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="created_predictions"
    )

    class Meta:
        db_table = "claim_ai_prediction"
        verbose_name = _("Claim Prediction")
        verbose_name_plural = _("Claim Predictions")
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["claim_id", "prediction_type"]),
            models.Index(fields=["ai_model", "created_at"]),
        ]

    def __str__(self):
        return f"Prediction {self.id} for claim {self.claim_id}"


class ClaimAIFeedback(models.Model):
    """
    Human feedback on AI predictions for continuous learning
    """

    FEEDBACK_TYPES = [
        ("correct", _("Correct Prediction")),
        ("incorrect", _("Incorrect Prediction")),
        ("partially_correct", _("Partially Correct")),
        ("uncertain", _("Uncertain")),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Related prediction
    prediction = models.OneToOneField(
        ClaimPrediction, on_delete=models.CASCADE, related_name="feedback"
    )

    # Feedback details
    feedback_type = models.CharField(max_length=20, choices=FEEDBACK_TYPES)
    feedback_notes = models.TextField(blank=True)
    actual_outcome = models.CharField(max_length=100, blank=True)

    # Quality metrics
    accuracy_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)], null=True, blank=True
    )
    usefulness_rating = models.IntegerField(
        validators=[MinValueValidator(1), MaxValueValidator(5)], null=True, blank=True
    )

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Relationships
    created_by = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="created_feedback"
    )

    class Meta:
        db_table = "claim_ai_feedback"
        verbose_name = _("AI Feedback")
        verbose_name_plural = _("AI Feedback")
        ordering = ["-created_at"]

    def __str__(self):
        return f"Feedback for prediction {self.prediction.id}"


class ClaimAITrainingData(models.Model):
    """
    Training data for AI models
    """

    DATA_TYPES = [
        ("training", _("Training Data")),
        ("validation", _("Validation Data")),
        ("test", _("Test Data")),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Data details
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    data_type = models.CharField(max_length=20, choices=DATA_TYPES)

    # File storage
    data_file = models.FileField(upload_to="training_data/")
    data_size = models.IntegerField(help_text=_("Number of records"))

    # Data schema
    feature_columns = models.JSONField(default=list)
    target_columns = models.JSONField(default=list)

    # Metadata
    source = models.CharField(max_length=255, blank=True)
    version = models.CharField(max_length=50)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Relationships
    created_by = models.ForeignKey(
        User, on_delete=models.SET_NULL, null=True, related_name="created_training_data"
    )

    class Meta:
        db_table = "claim_ai_training_data"
        verbose_name = _("Training Data")
        verbose_name_plural = _("Training Data")
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.name} ({self.get_data_type_display()})"


class ClaimAIModelPerformance(models.Model):
    """
    Performance tracking for AI models over time
    """

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)

    # Model reference
    ai_model = models.ForeignKey(
        ClaimAIModel, on_delete=models.CASCADE, related_name="performance_records"
    )

    # Performance metrics
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    auc_score = models.FloatField()

    # Additional metrics
    false_positive_rate = models.FloatField()
    false_negative_rate = models.FloatField()
    true_positive_rate = models.FloatField()
    true_negative_rate = models.FloatField()

    # Evaluation context
    evaluation_date = models.DateTimeField()
    test_data_size = models.IntegerField()
    evaluation_notes = models.TextField(blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "claim_ai_model_performance"
        verbose_name = _("Model Performance")
        verbose_name_plural = _("Model Performance")
        ordering = ["-evaluation_date"]

    def __str__(self):
        return f"Performance for {self.ai_model.name} on {self.evaluation_date.date()}"
