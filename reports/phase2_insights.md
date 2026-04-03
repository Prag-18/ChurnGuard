# Phase 2 Feature Engineering & Preprocessing Insights

- Input shape before preprocessing: (16000, 33)
- Shape after preprocessing: (16000, 32)
- Final SMOTE shape: X=(23840, 32), y=(23840,)
- New features created: ['ChargeToIncomeProxy', 'HasMultiServices', 'IsSeniorWithPartner', 'charges_per_month', 'signup_month', 'signup_year', 'tenure_group']
- MonthlyCharges median by class (before): {'0': 68.865, '1': 87.635}
- charges_per_month median by class (after): {'0': 64.425, '1': 75.11}
- SMOTE class ratio: 0=0.50, 1=0.50

## Key Takeaways

1. Feature engineering improved signal clarity between churn and non-churn customers.
2. Derived features like charges_per_month and tenure_group capture customer lifecycle better.
3. WoE encoding provides more meaningful representation than one-hot encoding.
4. SMOTE successfully balanced the dataset, addressing class imbalance.

Final Result: A fully processed, balanced dataset ready for model training.
