"""
Django signals for OpenIMIS Claim AI module
"""

from django.db.models.signals import post_save, pre_save, post_delete
from django.dispatch import receiver
from django.utils import timezone
from django.db import models
import logging

from .models import (
    ClaimAIModel,
    ClaimPrediction,
    ClaimAIFeedback,
    ClaimAIModelPerformance,
)

logger = logging.getLogger(__name__)


@receiver(post_save, sender=ClaimAIModel)
def handle_ai_model_save(sender, instance, created, **kwargs):
    """
    Handle AI model save events
    """
    if created:
        logger.info(f"New AI model created: {instance.name} ({instance.model_type})")
    else:
        logger.info(f"AI model updated: {instance.name}")


@receiver(post_save, sender=ClaimPrediction)
def handle_prediction_save(sender, instance, created, **kwargs):
    """
    Handle prediction save events
    """
    if created:
        logger.info(f"New prediction created for claim {instance.claim_id}")

        # Update model statistics
        try:
            model = instance.ai_model
            total_predictions = model.predictions.count()
            avg_confidence = (
                model.predictions.aggregate(
                    avg_confidence=models.Avg("confidence_score")
                )["avg_confidence"]
                or 0.0
            )

            # Log prediction statistics
            logger.info(
                f"Model {model.name}: {total_predictions} predictions, "
                f"avg confidence: {avg_confidence:.2f}"
            )
        except Exception as e:
            logger.error(f"Error updating model statistics: {e}")


@receiver(post_save, sender=ClaimAIFeedback)
def handle_feedback_save(sender, instance, created, **kwargs):
    """
    Handle feedback save events
    """
    if created:
        logger.info(f"New feedback received for prediction {instance.prediction.id}")

        # Update model performance based on feedback
        try:
            prediction = instance.prediction
            model = prediction.ai_model

            # Calculate feedback statistics
            feedback_stats = model.predictions.filter(feedback__isnull=False).aggregate(
                total_feedback=models.Count("feedback"),
                avg_accuracy_rating=models.Avg("feedback__accuracy_rating"),
                avg_usefulness_rating=models.Avg("feedback__usefulness_rating"),
            )

            logger.info(
                f"Model {model.name} feedback stats: "
                f"{feedback_stats['total_feedback']} feedback, "
                f"avg accuracy: {feedback_stats['avg_accuracy_rating']:.2f}, "
                f"avg usefulness: {feedback_stats['avg_usefulness_rating']:.2f}"
            )
        except Exception as e:
            logger.error(f"Error updating feedback statistics: {e}")


@receiver(post_save, sender=ClaimAIModelPerformance)
def handle_performance_save(sender, instance, created, **kwargs):
    """
    Handle performance save events
    """
    if created:
        logger.info(f"New performance record for model {instance.ai_model.name}")

        # Update model with latest performance metrics
        try:
            model = instance.ai_model
            model.accuracy = instance.accuracy
            model.precision = instance.precision
            model.recall = instance.recall
            model.f1_score = instance.f1_score
            model.auc_score = instance.auc_score
            model.save(
                update_fields=[
                    "accuracy",
                    "precision",
                    "recall",
                    "f1_score",
                    "auc_score",
                ]
            )

            logger.info(f"Updated model {model.name} with latest performance metrics")
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")


@receiver(pre_save, sender=ClaimAIModel)
def handle_model_status_change(sender, instance, **kwargs):
    """
    Handle model status changes
    """
    if instance.pk:
        try:
            old_instance = ClaimAIModel.objects.get(pk=instance.pk)
            if old_instance.status != instance.status:
                logger.info(
                    f"Model {instance.name} status changed from "
                    f"{old_instance.status} to {instance.status}"
                )

                # If model is being activated, update trained_at timestamp
                if instance.status == "active" and old_instance.status != "active":
                    instance.trained_at = timezone.now()

        except ClaimAIModel.DoesNotExist:
            pass


@receiver(post_delete, sender=ClaimAIModel)
def handle_model_deletion(sender, instance, **kwargs):
    """
    Handle model deletion events
    """
    logger.info(f"AI model deleted: {instance.name} ({instance.model_type})")


@receiver(post_delete, sender=ClaimPrediction)
def handle_prediction_deletion(sender, instance, **kwargs):
    """
    Handle prediction deletion events
    """
    logger.info(f"Prediction deleted for claim {instance.claim_id}")


# Custom signals for advanced functionality
from django.dispatch import Signal

# Signal emitted when a new prediction is made
prediction_made = Signal()

# Signal emitted when feedback is provided
feedback_provided = Signal()

# Signal emitted when model performance is evaluated
performance_evaluated = Signal()

# Signal emitted when model training starts/completes
model_training_started = Signal()
model_training_completed = Signal()


@receiver(prediction_made)
def handle_prediction_made(sender, prediction, **kwargs):
    """
    Handle custom prediction made signal
    """
    logger.info(
        f"Custom prediction signal: {prediction.claim_id} - "
        f"{prediction.prediction_type} ({prediction.confidence_score:.2f})"
    )


@receiver(feedback_provided)
def handle_feedback_provided(sender, feedback, **kwargs):
    """
    Handle custom feedback provided signal
    """
    logger.info(
        f"Custom feedback signal: {feedback.feedback_type} - "
        f"accuracy: {feedback.accuracy_rating}, "
        f"usefulness: {feedback.usefulness_rating}"
    )


@receiver(performance_evaluated)
def handle_performance_evaluated(sender, performance, **kwargs):
    """
    Handle custom performance evaluation signal
    """
    logger.info(
        f"Custom performance signal: {performance.ai_model.name} - "
        f"accuracy: {performance.accuracy:.2f}"
    )


@receiver(model_training_started)
def handle_training_started(sender, model, **kwargs):
    """
    Handle model training started signal
    """
    logger.info(f"Model training started: {model.name}")


@receiver(model_training_completed)
def handle_training_completed(sender, model, performance_metrics, **kwargs):
    """
    Handle model training completed signal
    """
    logger.info(
        f"Model training completed: {model.name} - "
        f"accuracy: {performance_metrics.get('accuracy', 0):.2f}"
    )
