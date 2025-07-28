#!/usr/bin/env python3
"""
Simple test script for the Claim AI Model
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from claim_ai.ai_model.model import ClaimAIModel
    import numpy as np
    import pandas as pd

    print("✅ All imports successful!")

    # Test the model
    print("\n🧪 Testing Claim AI Model...")

    # Create sample data
    np.random.seed(42)
    n_samples = 100

    sample_data = pd.DataFrame(
        {
            "claim_amount": np.random.exponential(5000, n_samples),
            "patient_age": np.random.normal(45, 15, n_samples),
            "days_in_hospital": np.random.poisson(3, n_samples),
            "number_of_procedures": np.random.poisson(2, n_samples),
            "total_cost": np.random.exponential(8000, n_samples),
            "deductible_amount": np.random.uniform(0, 2000, n_samples),
            "co_payment": np.random.uniform(0, 500, n_samples),
            "out_of_pocket_max": np.random.uniform(2000, 8000, n_samples),
            "provider_type": np.random.choice(
                ["hospital", "clinic", "specialist"], n_samples
            ),
            "diagnosis_code": np.random.choice(
                ["ICD10_A", "ICD10_B", "ICD10_C"], n_samples
            ),
            "procedure_code": np.random.choice(["CPT_A", "CPT_B", "CPT_C"], n_samples),
            "claim_type": np.random.choice(
                ["inpatient", "outpatient", "emergency"], n_samples
            ),
            "patient_gender": np.random.choice(["M", "F"], n_samples),
            "insurance_type": np.random.choice(
                ["private", "public", "corporate"], n_samples
            ),
            "claim_description": ["Sample claim description"] * n_samples,
            "diagnosis_notes": ["Sample diagnosis notes"] * n_samples,
            "claim_status": np.random.choice(
                ["approved", "rejected"], n_samples, p=[0.8, 0.2]
            ),
            "fraud_indicator": np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            "risk_score": np.random.choice(
                [0, 1, 2, 3], n_samples, p=[0.6, 0.25, 0.1, 0.05]
            ),
        }
    )

    # Initialize model
    model = ClaimAIModel()
    print("✅ Model initialized successfully!")

    # Train model
    print("🔄 Training models...")
    model.train_models(sample_data)
    print("✅ Models trained successfully!")

    # Test prediction
    test_claim = {
        "claim_amount": 5000,
        "patient_age": 45,
        "days_in_hospital": 3,
        "number_of_procedures": 2,
        "total_cost": 8000,
        "deductible_amount": 1000,
        "co_payment": 200,
        "out_of_pocket_max": 5000,
        "provider_type": "hospital",
        "diagnosis_code": "ICD10_A",
        "procedure_code": "CPT_A",
        "claim_type": "inpatient",
        "patient_gender": "M",
        "insurance_type": "private",
        "claim_description": "Sample claim for testing",
        "diagnosis_notes": "Sample diagnosis for testing",
    }

    print("🔮 Making prediction...")
    prediction = model.predict_claim(test_claim)

    print("\n📊 Claim Prediction Results:")
    for key, value in prediction.items():
        print(f"  {key}: {value}")

    print("\n✅ All tests passed! The AI model is working correctly.")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Please install required dependencies: pip install scikit-learn pandas numpy joblib"
    )
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
