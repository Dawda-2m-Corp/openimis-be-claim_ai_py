"""
Django views for OpenIMIS Claim AI module
"""

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.paginator import Paginator
from django.db.models import Q, Avg, Count
from django.utils.decorators import method_decorator
from django.views import View
from django.utils.translation import gettext_lazy as _
import json
import logging
from datetime import datetime, timedelta

from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import ModelViewSet
from rest_framework.pagination import PageNumberPagination

from .models import (
    ClaimAIModel,
    ClaimPrediction,
    ClaimAIFeedback,
    ClaimAITrainingData,
    ClaimAIModelPerformance,
)
from .ai_model.model import ClaimAIModel as AIModelClass
from .serializers import (
    ClaimAIModelSerializer,
    ClaimPredictionSerializer,
    ClaimAIFeedbackSerializer,
    ClaimAITrainingDataSerializer,
    ClaimAIModelPerformanceSerializer,
)

logger = logging.getLogger(__name__)


class ClaimAIModelViewSet(ModelViewSet):
    """ViewSet for AI models"""

    queryset = ClaimAIModel.objects.all()
    serializer_class = ClaimAIModelSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = PageNumberPagination

    def get_queryset(self):
        queryset = ClaimAIModel.objects.all()

        # Filter by model type
        model_type = self.request.query_params.get("model_type", None)
        if model_type:
            queryset = queryset.filter(model_type=model_type)

        # Filter by status
        status = self.request.query_params.get("status", None)
        if status:
            queryset = queryset.filter(status=status)

        # Search by name
        search = self.request.query_params.get("search", None)
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search)
            )

        return queryset.order_by("-created_at")


class ClaimPredictionViewSet(ModelViewSet):
    """ViewSet for claim predictions"""

    queryset = ClaimPrediction.objects.all()
    serializer_class = ClaimPredictionSerializer
    permission_classes = [IsAuthenticated]
    pagination_class = PageNumberPagination

    def get_queryset(self):
        queryset = ClaimPrediction.objects.select_related("ai_model")

        # Filter by claim ID
        claim_id = self.request.query_params.get("claim_id", None)
        if claim_id:
            queryset = queryset.filter(claim_id=claim_id)

        # Filter by prediction type
        prediction_type = self.request.query_params.get("prediction_type", None)
        if prediction_type:
            queryset = queryset.filter(prediction_type=prediction_type)

        # Filter by confidence level
        confidence_level = self.request.query_params.get("confidence_level", None)
        if confidence_level:
            queryset = queryset.filter(confidence_level=confidence_level)

        # Filter by AI model
        ai_model = self.request.query_params.get("ai_model", None)
        if ai_model:
            queryset = queryset.filter(ai_model_id=ai_model)

        return queryset.order_by("-created_at")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def predict_claim(request):
    """
    Make AI prediction for a claim
    """
    try:
        data = request.data

        # Validate required fields
        required_fields = ["claim_id", "claim_data"]
        for field in required_fields:
            if field not in data:
                return Response(
                    {"error": f"Missing required field: {field}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        claim_id = data["claim_id"]
        claim_data = data["claim_data"]
        model_id = data.get("model_id")

        # Get active AI model
        if model_id:
            ai_model = get_object_or_404(ClaimAIModel, id=model_id, status="active")
        else:
            # Get the most recent active model
            ai_model = (
                ClaimAIModel.objects.filter(status="active")
                .order_by("-created_at")
                .first()
            )
            if not ai_model:
                return Response(
                    {"error": "No active AI model found"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Initialize AI model class
        model_instance = AIModelClass()

        # Load trained model if available
        if ai_model.model_file:
            try:
                model_instance.load_models(ai_model.model_file.path)
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return Response(
                    {"error": "Failed to load AI model"},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

        # Make prediction
        prediction_result = model_instance.predict_claim(claim_data)

        # Create prediction record
        prediction = ClaimPrediction.objects.create(
            claim_id=claim_id,
            prediction_type=data.get("prediction_type", "approval"),
            prediction_value=prediction_result.get("approval_probability", 0.0),
            confidence_score=prediction_result.get("confidence_score", 0.0),
            confidence_level=_get_confidence_level(
                prediction_result.get("confidence_score", 0.0)
            ),
            ai_model=ai_model,
            input_features=claim_data,
            prediction_probabilities=prediction_result.get("probabilities", {}),
            explanation=prediction_result.get("explanation", ""),
            feature_importance=prediction_result.get("feature_importance", {}),
            created_by=request.user,
        )

        # Serialize response
        serializer = ClaimPredictionSerializer(prediction)

        return Response(
            {
                "prediction": serializer.data,
                "ai_model": {
                    "id": ai_model.id,
                    "name": ai_model.name,
                    "version": ai_model.version,
                },
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return Response(
            {"error": "Failed to make prediction"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def train_model(request):
    """
    Train a new AI model
    """
    try:
        data = request.data

        # Validate required fields
        required_fields = ["name", "model_type", "training_data_id"]
        for field in required_fields:
            if field not in data:
                return Response(
                    {"error": f"Missing required field: {field}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Get training data
        training_data = get_object_or_404(
            ClaimAITrainingData, id=data["training_data_id"]
        )

        # Create AI model record
        ai_model = ClaimAIModel.objects.create(
            name=data["name"],
            model_type=data["model_type"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            algorithm=data.get("algorithm", "RandomForest"),
            hyperparameters=data.get("hyperparameters", {}),
            status="training",
            created_by=request.user,
        )

        # Initialize AI model class
        model_instance = AIModelClass()

        # Load training data (this would need to be implemented based on your data format)
        # training_dataset = load_training_data(training_data.data_file.path)

        # Train model
        # model_instance.train_models(training_dataset)

        # Update model status
        ai_model.status = "active"
        ai_model.trained_at = datetime.now()
        ai_model.save()

        serializer = ClaimAIModelSerializer(ai_model)

        return Response(
            {
                "message": "Model training completed successfully",
                "model": serializer.data,
            },
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.error(f"Training error: {e}")
        return Response(
            {"error": "Failed to train model"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def provide_feedback(request):
    """
    Provide feedback on AI prediction
    """
    try:
        data = request.data

        # Validate required fields
        required_fields = ["prediction_id", "feedback_type"]
        for field in required_fields:
            if field not in data:
                return Response(
                    {"error": f"Missing required field: {field}"},
                    status=status.HTTP_400_BAD_REQUEST,
                )

        # Get prediction
        prediction = get_object_or_404(ClaimPrediction, id=data["prediction_id"])

        # Create or update feedback
        feedback, created = ClaimAIFeedback.objects.get_or_create(
            prediction=prediction,
            defaults={
                "feedback_type": data["feedback_type"],
                "feedback_notes": data.get("feedback_notes", ""),
                "actual_outcome": data.get("actual_outcome", ""),
                "accuracy_rating": data.get("accuracy_rating"),
                "usefulness_rating": data.get("usefulness_rating"),
                "created_by": request.user,
            },
        )

        if not created:
            # Update existing feedback
            feedback.feedback_type = data["feedback_type"]
            feedback.feedback_notes = data.get("feedback_notes", "")
            feedback.actual_outcome = data.get("actual_outcome", "")
            feedback.accuracy_rating = data.get("accuracy_rating")
            feedback.usefulness_rating = data.get("usefulness_rating")
            feedback.save()

        serializer = ClaimAIFeedbackSerializer(feedback)

        return Response(
            {"message": "Feedback provided successfully", "feedback": serializer.data},
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return Response(
            {"error": "Failed to provide feedback"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def model_performance(request, model_id):
    """
    Get performance metrics for a specific model
    """
    try:
        ai_model = get_object_or_404(ClaimAIModel, id=model_id)

        # Get performance records
        performance_records = ClaimAIModelPerformance.objects.filter(
            ai_model=ai_model
        ).order_by("-evaluation_date")

        # Calculate statistics
        total_predictions = ClaimPrediction.objects.filter(ai_model=ai_model).count()
        avg_confidence = (
            ClaimPrediction.objects.filter(ai_model=ai_model).aggregate(
                avg_confidence=Avg("confidence_score")
            )["avg_confidence"]
            or 0.0
        )

        # Get feedback statistics
        feedback_stats = ClaimAIFeedback.objects.filter(
            prediction__ai_model=ai_model
        ).aggregate(
            total_feedback=Count("id"),
            avg_accuracy_rating=Avg("accuracy_rating"),
            avg_usefulness_rating=Avg("usefulness_rating"),
        )

        serializer = ClaimAIModelPerformanceSerializer(performance_records, many=True)

        return Response(
            {
                "model": ClaimAIModelSerializer(ai_model).data,
                "performance_records": serializer.data,
                "statistics": {
                    "total_predictions": total_predictions,
                    "avg_confidence": avg_confidence,
                    "total_feedback": feedback_stats["total_feedback"] or 0,
                    "avg_accuracy_rating": feedback_stats["avg_accuracy_rating"] or 0.0,
                    "avg_usefulness_rating": feedback_stats["avg_usefulness_rating"]
                    or 0.0,
                },
            }
        )

    except Exception as e:
        logger.error(f"Performance error: {e}")
        return Response(
            {"error": "Failed to get performance data"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def dashboard_stats(request):
    """
    Get dashboard statistics
    """
    try:
        # Get date range
        days = int(request.query_params.get("days", 30))
        start_date = datetime.now() - timedelta(days=days)

        # Model statistics
        total_models = ClaimAIModel.objects.count()
        active_models = ClaimAIModel.objects.filter(status="active").count()

        # Prediction statistics
        total_predictions = ClaimPrediction.objects.filter(
            created_at__gte=start_date
        ).count()

        predictions_by_type = (
            ClaimPrediction.objects.filter(created_at__gte=start_date)
            .values("prediction_type")
            .annotate(count=Count("id"))
        )

        # Feedback statistics
        total_feedback = ClaimAIFeedback.objects.filter(
            created_at__gte=start_date
        ).count()

        avg_accuracy_rating = (
            ClaimAIFeedback.objects.filter(created_at__gte=start_date).aggregate(
                avg_rating=Avg("accuracy_rating")
            )["avg_rating"]
            or 0.0
        )

        return Response(
            {
                "models": {"total": total_models, "active": active_models},
                "predictions": {
                    "total": total_predictions,
                    "by_type": list(predictions_by_type),
                },
                "feedback": {
                    "total": total_feedback,
                    "avg_accuracy_rating": avg_accuracy_rating,
                },
            }
        )

    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return Response(
            {"error": "Failed to get dashboard statistics"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


def _get_confidence_level(confidence_score):
    """Convert confidence score to level"""
    if confidence_score >= 0.8:
        return "high"
    elif confidence_score >= 0.6:
        return "medium"
    else:
        return "low"


# Legacy view functions for backward compatibility
@csrf_exempt
@require_http_methods(["POST"])
def legacy_predict_claim(request):
    """Legacy endpoint for claim prediction"""
    try:
        data = json.loads(request.body)
        # Convert to new API format and call predict_claim
        # This is a simplified version
        return JsonResponse({"status": "success", "message": "Use new API endpoint"})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)


@login_required
def model_dashboard(request):
    """Dashboard view for AI models"""
    models = ClaimAIModel.objects.all().order_by("-created_at")
    predictions = ClaimPrediction.objects.all().order_by("-created_at")[:10]

    context = {
        "models": models,
        "recent_predictions": predictions,
    }

    return render(request, "claim_ai/dashboard.html", context)
