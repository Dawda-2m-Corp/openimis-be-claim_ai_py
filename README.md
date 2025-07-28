# OpenIMIS Claim AI

An AI-powered claim processing system for OpenIMIS (Open Insurance Management Information System).

## Overview

This Django application provides intelligent claim processing capabilities using AI models to automate and enhance the claim validation and processing workflow.

## Features

- AI-powered claim analysis and validation
- Automated claim processing workflows
- Integration with OpenIMIS core system
- Machine learning model management
- RESTful API endpoints for claim processing

## Project Structure

```
openimis_claim_ai/
├── claim_ai/              # Main Django app
│   ├── ai_model/          # AI model components
│   ├── migrations/        # Database migrations
│   ├── admin.py           # Django admin configuration
│   ├── models.py          # Data models
│   ├── views.py           # View logic
│   └── tests.py           # Test cases
├── setup.py               # Package configuration
└── venv/                  # Virtual environment
```

## Setup

### Prerequisites

- Python 3.10+
- Django 5.2+
- Virtual environment

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd openimis_claim_ai
```

2. Create and activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run migrations:

```bash
python manage.py migrate
```

5. Start the development server:

```bash
python manage.py runserver
```

## Usage

### API Endpoints

- `/api/claims/` - Claim processing endpoints
- `/admin/` - Django admin interface

### AI Model Integration

The AI model components are located in `claim_ai/ai_model/` and provide:

- Claim validation algorithms
- Fraud detection capabilities
- Automated decision making

## Development

### Running Tests

```bash
python manage.py test
```

### Code Style

Follow PEP 8 guidelines for Python code formatting.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Add your license information here]

## Contact

[Add contact information here]
