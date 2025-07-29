"""
URL configuration for OpenIMIS Claim AI module
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register(r"models", views.ClaimAIModelViewSet, basename="claim-ai-model")
router.register(
    r"predictions", views.ClaimPredictionViewSet, basename="claim-prediction"
)

# API URL patterns
api_urlpatterns = [
    # ViewSet endpoints
    path("", include(router.urls)),
    # Custom API endpoints
    path("predict/", views.predict_claim, name="predict-claim"),
    path("train/", views.train_model, name="train-model"),
    path("feedback/", views.provide_feedback, name="provide-feedback"),
    path(
        "models/<uuid:model_id>/performance/",
        views.model_performance,
        name="model-performance",
    ),
    path("dashboard/stats/", views.dashboard_stats, name="dashboard-stats"),
]

# Legacy URL patterns for backward compatibility
legacy_urlpatterns = [
    path("legacy/predict/", views.legacy_predict_claim, name="legacy-predict-claim"),
]

# Django view URL patterns
django_urlpatterns = [
    path("dashboard/", views.model_dashboard, name="model-dashboard"),
]

# Combined URL patterns
urlpatterns = [
    # API endpoints
    path("api/v1/", include(api_urlpatterns)),
    # Legacy endpoints
    path("legacy/", include(legacy_urlpatterns)),
    # Django views
    path("", include(django_urlpatterns)),
]
