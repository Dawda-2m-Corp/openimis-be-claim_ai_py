"""
Django admin configuration for OpenIMIS Claim AI module
"""

from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.db.models import Avg, Count
from .models import (
    ClaimAIModel,
    ClaimPrediction,
    ClaimAIFeedback,
    ClaimAITrainingData,
    ClaimAIModelPerformance,
)


@admin.register(ClaimAIModel)
class ClaimAIModelAdmin(admin.ModelAdmin):
    """Admin interface for AI models"""

    list_display = [
        "name",
        "model_type",
        "version",
        "status",
        "accuracy",
        "created_at",
        "trained_at",
        "created_by",
    ]
    list_filter = ["model_type", "status", "created_at", "trained_at"]
    search_fields = ["name", "description", "algorithm"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            _("Basic Information"),
            {"fields": ("id", "name", "model_type", "version", "status")},
        ),
        (
            _("Model Details"),
            {"fields": ("description", "algorithm", "hyperparameters")},
        ),
        (
            _("Performance Metrics"),
            {
                "fields": ("accuracy", "precision", "recall", "f1_score", "auc_score"),
                "classes": ("collapse",),
            },
        ),
        (
            _("Model Files"),
            {
                "fields": ("model_file", "scaler_file", "encoder_file"),
                "classes": ("collapse",),
            },
        ),
        (
            _("Timestamps"),
            {
                "fields": ("created_at", "updated_at", "trained_at"),
                "classes": ("collapse",),
            },
        ),
        (
            _("Relationships"),
            {"fields": ("created_by", "updated_by"), "classes": ("collapse",)},
        ),
    )

    def get_queryset(self, request):
        """Add performance statistics"""
        queryset = super().get_queryset(request)
        return queryset.annotate(
            prediction_count=Count("predictions"),
            avg_confidence=Avg("predictions__confidence_score"),
        )

    def prediction_count(self, obj):
        """Display prediction count"""
        return obj.prediction_count

    prediction_count.short_description = _("Predictions")

    def avg_confidence(self, obj):
        """Display average confidence"""
        if obj.avg_confidence:
            return f"{obj.avg_confidence:.2f}"
        return "-"

    avg_confidence.short_description = _("Avg Confidence")


@admin.register(ClaimPrediction)
class ClaimPredictionAdmin(admin.ModelAdmin):
    """Admin interface for claim predictions"""

    list_display = [
        "id",
        "claim_id",
        "prediction_type",
        "prediction_value",
        "confidence_score",
        "confidence_level",
        "ai_model",
        "created_at",
    ]
    list_filter = ["prediction_type", "confidence_level", "ai_model", "created_at"]
    search_fields = ["claim_id", "explanation"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            _("Basic Information"),
            {"fields": ("id", "claim_id", "prediction_type", "ai_model")},
        ),
        (
            _("Prediction Details"),
            {"fields": ("prediction_value", "confidence_score", "confidence_level")},
        ),
        (
            _("Model Data"),
            {
                "fields": (
                    "input_features",
                    "prediction_probabilities",
                    "feature_importance",
                ),
                "classes": ("collapse",),
            },
        ),
        (_("Explanation"), {"fields": ("explanation",), "classes": ("collapse",)}),
        (
            _("Timestamps"),
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
        (_("Relationships"), {"fields": ("created_by",), "classes": ("collapse",)}),
    )

    def get_queryset(self, request):
        """Add feedback statistics"""
        queryset = super().get_queryset(request)
        return queryset.annotate(feedback_count=Count("feedback"))

    def feedback_count(self, obj):
        """Display feedback count"""
        return obj.feedback_count

    feedback_count.short_description = _("Feedback")


@admin.register(ClaimAIFeedback)
class ClaimAIFeedbackAdmin(admin.ModelAdmin):
    """Admin interface for AI feedback"""

    list_display = [
        "id",
        "prediction",
        "feedback_type",
        "accuracy_rating",
        "usefulness_rating",
        "created_by",
        "created_at",
    ]
    list_filter = [
        "feedback_type",
        "accuracy_rating",
        "usefulness_rating",
        "created_at",
    ]
    search_fields = ["feedback_notes", "actual_outcome"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (_("Basic Information"), {"fields": ("id", "prediction", "feedback_type")}),
        (_("Feedback Details"), {"fields": ("feedback_notes", "actual_outcome")}),
        (_("Quality Ratings"), {"fields": ("accuracy_rating", "usefulness_rating")}),
        (
            _("Timestamps"),
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
        (_("Relationships"), {"fields": ("created_by",), "classes": ("collapse",)}),
    )


@admin.register(ClaimAITrainingData)
class ClaimAITrainingDataAdmin(admin.ModelAdmin):
    """Admin interface for training data"""

    list_display = [
        "name",
        "data_type",
        "data_size",
        "version",
        "source",
        "created_by",
        "created_at",
    ]
    list_filter = ["data_type", "created_at"]
    search_fields = ["name", "description", "source"]
    readonly_fields = ["id", "created_at", "updated_at"]

    fieldsets = (
        (
            _("Basic Information"),
            {"fields": ("id", "name", "description", "data_type")},
        ),
        (
            _("Data Details"),
            {"fields": ("data_file", "data_size", "version", "source")},
        ),
        (
            _("Schema Information"),
            {"fields": ("feature_columns", "target_columns"), "classes": ("collapse",)},
        ),
        (
            _("Timestamps"),
            {"fields": ("created_at", "updated_at"), "classes": ("collapse",)},
        ),
        (_("Relationships"), {"fields": ("created_by",), "classes": ("collapse",)}),
    )


@admin.register(ClaimAIModelPerformance)
class ClaimAIModelPerformanceAdmin(admin.ModelAdmin):
    """Admin interface for model performance tracking"""

    list_display = [
        "ai_model",
        "evaluation_date",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "test_data_size",
    ]
    list_filter = ["ai_model", "evaluation_date"]
    search_fields = ["evaluation_notes"]
    readonly_fields = ["id", "created_at"]

    fieldsets = (
        (_("Basic Information"), {"fields": ("id", "ai_model", "evaluation_date")}),
        (
            _("Performance Metrics"),
            {"fields": ("accuracy", "precision", "recall", "f1_score", "auc_score")},
        ),
        (
            _("Additional Metrics"),
            {
                "fields": (
                    "false_positive_rate",
                    "false_negative_rate",
                    "true_positive_rate",
                    "true_negative_rate",
                ),
                "classes": ("collapse",),
            },
        ),
        (
            _("Evaluation Context"),
            {
                "fields": ("test_data_size", "evaluation_notes"),
                "classes": ("collapse",),
            },
        ),
        (_("Timestamps"), {"fields": ("created_at",), "classes": ("collapse",)}),
    )

    def get_queryset(self, request):
        """Add model information"""
        return super().get_queryset(request).select_related("ai_model")


# Custom admin site configuration
admin.site.site_header = _("OpenIMIS Claim AI Administration")
admin.site.site_title = _("Claim AI Admin")
admin.site.index_title = _("Claim AI Management")
