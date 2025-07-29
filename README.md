# OpenIMIS Claim AI

An AI-powered claim processing module for OpenIMIS (Open Insurance Management Information System) that provides intelligent claim analysis, fraud detection, and automated decision-making capabilities.

## Overview

This Django application integrates advanced machine learning algorithms with the OpenIMIS claim processing workflow to enhance efficiency, accuracy, and fraud detection capabilities. The module follows openIMIS architecture patterns and provides both REST API endpoints and Django admin interfaces.

## Features

### Core AI Capabilities

- **Fraud Detection**: Advanced anomaly detection using Isolation Forest and One-Class SVM
- **Claim Classification**: Automated approval/rejection prediction using Random Forest
- **Risk Scoring**: Logistic regression-based risk assessment
- **Quality Assessment**: AI-powered claim quality evaluation
- **Anomaly Detection**: Real-time detection of unusual claim patterns

### Management Features

- **Model Management**: Complete lifecycle management of AI models
- **Performance Tracking**: Continuous monitoring of model performance
- **Feedback System**: Human-in-the-loop feedback for continuous learning
- **Training Data Management**: Secure handling of training datasets
- **Dashboard Analytics**: Comprehensive analytics and reporting

### Integration Features

- **REST API**: Full REST API for external integrations
- **Django Admin**: Complete admin interface for management
- **OpenIMIS Integration**: Seamless integration with openIMIS core
- **GraphQL Support**: GraphQL endpoints for flexible data querying
- **Authentication**: Secure authentication and authorization

## Project Structure

```
openimis_claim_ai/
├── claim_ai/                    # Main Django app
│   ├── ai_model/               # AI model components
│   │   ├── __init__.py
│   │   └── model.py           # Core AI model implementation
│   ├── migrations/             # Database migrations
│   ├── admin.py               # Django admin configuration
│   ├── models.py              # Data models
│   ├── views.py               # View logic and API endpoints
│   ├── serializers.py         # REST API serializers
│   ├── urls.py                # URL configuration
│   ├── tests.py               # Comprehensive test suite
│   └── apps.py                # Django app configuration
├── setup.py                   # Package configuration
├── requirements.txt            # Dependencies
├── test_model.py              # Model testing script
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.8+
- Django 4.2+
- PostgreSQL (recommended) or SQLite
- Redis (for caching and Celery)
- Virtual environment

### Quick Start

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd openimis_claim_ai
   ```

2. **Create and activate virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Django settings**:

   ```python
   # Add to your Django settings
   INSTALLED_APPS = [
       # ... other apps
       'claim_ai',
   ]

   # Configure media files for model storage
   MEDIA_URL = '/media/'
   MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
   ```

5. **Run migrations**:

   ```bash
   python manage.py makemigrations claim_ai
   python manage.py migrate
   ```

6. **Create superuser**:

   ```bash
   python manage.py createsuperuser
   ```

7. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

## Usage

### API Endpoints

#### Authentication

All API endpoints require authentication. Use Django REST Framework's authentication:

```bash
# Include Authorization header
Authorization: Token <your-token>
```

#### Core Endpoints

**Predict Claim**

```bash
POST /api/v1/predict/
Content-Type: application/json

{
    "claim_id": "CLAIM-001",
    "claim_data": {
        "amount": 5000,
        "patient_age": 45,
        "days_in_hospital": 3,
        "provider_type": "hospital",
        "diagnosis_code": "ICD10_A",
        "procedure_code": "CPT_A"
    },
    "prediction_type": "fraud"
}
```

**Train Model**

```bash
POST /api/v1/train/
Content-Type: application/json

{
    "name": "Fraud Detection Model v2.0",
    "model_type": "fraud_detection",
    "training_data_id": "uuid-of-training-data",
    "version": "2.0.0",
    "description": "Updated fraud detection model",
    "algorithm": "RandomForest",
    "hyperparameters": {
        "n_estimators": 200,
        "max_depth": 15
    }
}
```

**Provide Feedback**

```bash
POST /api/v1/feedback/
Content-Type: application/json

{
    "prediction_id": "uuid-of-prediction",
    "feedback_type": "correct",
    "feedback_notes": "Prediction was accurate",
    "actual_outcome": "fraud_detected",
    "accuracy_rating": 5,
    "usefulness_rating": 4
}
```

**Get Dashboard Statistics**

```bash
GET /api/v1/dashboard/stats/?days=30
```

#### ViewSet Endpoints

**AI Models**

```bash
GET    /api/v1/models/                    # List models
POST   /api/v1/models/                    # Create model
GET    /api/v1/models/{id}/               # Get model
PUT    /api/v1/models/{id}/               # Update model
DELETE /api/v1/models/{id}/               # Delete model
```

**Predictions**

```bash
GET    /api/v1/predictions/               # List predictions
GET    /api/v1/predictions/{id}/          # Get prediction
PUT    /api/v1/predictions/{id}/          # Update prediction
DELETE /api/v1/predictions/{id}/          # Delete prediction
```

### Django Admin Interface

Access the admin interface at `/admin/` to manage:

- **AI Models**: Create, update, and monitor AI models
- **Predictions**: View and manage claim predictions
- **Feedback**: Review and manage human feedback
- **Training Data**: Upload and manage training datasets
- **Performance**: Track model performance metrics

### AI Model Integration

The AI model components are located in `claim_ai/ai_model/` and provide:

```python
from claim_ai.ai_model.model import ClaimAIModel

# Initialize model
model = ClaimAIModel()

# Train models
model.train_models(training_data)

# Make predictions
prediction = model.predict_claim(claim_data)

# Save/load models
model.save_models('path/to/models')
model.load_models('path/to/models')
```

## Configuration

### Model Configuration

Configure AI models through Django admin or API:

```python
# Example model configuration
model_config = {
    'name': 'Fraud Detection Model',
    'model_type': 'fraud_detection',
    'algorithm': 'RandomForest',
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
}
```

### Performance Monitoring

Track model performance automatically:

```python
# Performance metrics are automatically calculated
performance_metrics = {
    'accuracy': 0.85,
    'precision': 0.82,
    'recall': 0.88,
    'f1_score': 0.85,
    'auc_score': 0.87
}
```

## Development

### Running Tests

```bash
# Run all tests
python manage.py test claim_ai

# Run specific test classes
python manage.py test claim_ai.tests.ClaimAIModelTestCase

# Run with coverage
coverage run --source='.' manage.py test claim_ai
coverage report
```

### Code Style

Follow PEP 8 guidelines and use the provided tools:

```bash
# Format code
black claim_ai/

# Check code style
flake8 claim_ai/

# Sort imports
isort claim_ai/

# Type checking
mypy claim_ai/
```

### Testing the AI Model

Use the provided test script:

```bash
python test_model.py
```

This will test the AI model with sample data and verify all components are working correctly.

## API Documentation

### Authentication

All API endpoints require authentication. Use one of these methods:

1. **Token Authentication**:

   ```bash
   Authorization: Token <your-token>
   ```

2. **Session Authentication** (for browser access):
   ```bash
   # Include session cookie
   ```

### Response Formats

All API responses follow this format:

```json
{
  "status": "success",
  "data": {
    // Response data
  },
  "message": "Optional message"
}
```

### Error Handling

Errors follow this format:

```json
{
  "error": "Error message",
  "details": {
    // Additional error details
  }
}
```

## Deployment

### Production Setup

1. **Environment Variables**:

   ```bash
   export DJANGO_SETTINGS_MODULE=your_project.settings
   export SECRET_KEY=your-secret-key
   export DATABASE_URL=your-database-url
   ```

2. **Static Files**:

   ```bash
   python manage.py collectstatic
   ```

3. **Database Migration**:

   ```bash
   python manage.py migrate
   ```

4. **WSGI Configuration**:

   ```python
   # wsgi.py
   import os
   from django.core.wsgi import get_wsgi_application

   os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'your_project.settings')
   application = get_wsgi_application()
   ```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python manage.py collectstatic --noinput

EXPOSE 8000
CMD ["gunicorn", "your_project.wsgi:application", "--bind", "0.0.0.0:8000"]
```

## Contributing

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Run tests**:
   ```bash
   python manage.py test claim_ai
   ```
6. **Submit a pull request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages
- Follow openIMIS coding standards

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [OpenIMIS Documentation](https://openimis.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/openimis/openimis-be-claim_ai_py/issues)
- **Community**: [OpenIMIS Community](https://openimis.org/)

## Acknowledgments

- OpenIMIS Community
- Contributors and maintainers
- Healthcare organizations using OpenIMIS

## Changelog

### Version 1.0.0

- Initial release
- Core AI model functionality
- REST API endpoints
- Django admin interface
- Comprehensive test suite
- Documentation and deployment guides
