"""
Django app configuration for OpenIMIS Claim AI module
"""

from django.apps import AppConfig
from django.utils.translation import gettext_lazy as _


class ClaimAIConfig(AppConfig):
    """
    Configuration for the Claim AI Django app
    """

    default_auto_field = "django.db.models.BigAutoField"
    name = "claim_ai"
    verbose_name = _("Claim AI")

    def ready(self):
        """
        Initialize the app when Django starts
        """
        # Import signals to register them
        try:
            import claim_ai.signals
        except ImportError:
            pass

        # Import admin to register models
        try:
            import claim_ai.admin
        except ImportError:
            pass
