from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

class CustomerInput(BaseModel):
    # Demographics
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    
    # Services
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    MultipleLines: str = Field(..., pattern="^(No phone service|No|Yes)$")
    InternetService: str = Field(..., pattern="^(DSL|Fiber optic|No)$")
    OnlineSecurity: str = Field(..., pattern="^(No|Yes|No internet service)$")
    OnlineBackup: str = Field(..., pattern="^(No|Yes|No internet service)$")
    DeviceProtection: str = Field(..., pattern="^(No|Yes|No internet service)$")
    TechSupport: str = Field(..., pattern="^(No|Yes|No internet service)$")
    StreamingTV: str = Field(..., pattern="^(No|Yes|No internet service)$")
    StreamingMovies: str = Field(..., pattern="^(No|Yes|No internet service)$")
    
    # Account
    Contract: str = Field(..., pattern="^(Month-to-month|One year|Two year)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    PaymentMethod: str = Field(..., pattern="^(Electronic check|Mailed check|Bank transfer \(automatic\)|Credit card \(automatic\))$")
    tenure: int = Field(..., ge=0, le=72)
    MonthlyCharges: float = Field(..., ge=18.0, le=120.0)
    TotalCharges: float = Field(..., ge=0.0, le=10000.0)

    # Behavioral (Optional as basic UI might omit them)
    last_active_days: Optional[int] = None
    avg_monthly_usage: Optional[float] = None
    number_of_logins: Optional[int] = None
    complaints_count: Optional[int] = None
    customer_support_calls: Optional[int] = None
    late_payments: Optional[int] = None

    @model_validator(mode='after')
    def validate_services(self) -> 'CustomerInput':
        if self.PhoneService == 'No' and self.MultipleLines != 'No phone service':
            raise ValueError('If PhoneService is No, MultipleLines must be "No phone service"')
        
        net_deps = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection, 
                    self.TechSupport, self.StreamingTV, self.StreamingMovies]
        if self.InternetService == 'No':
            if any(val != 'No internet service' for val in net_deps):
                raise ValueError('If InternetService is No, all internet-dependent services must be "No internet service"')
        return self

class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput] = Field(..., max_length=1000)

class WhatIfInterventions(BaseModel):
    Contract: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TechSupport: Optional[str] = None

class WhatIfInput(BaseModel):
    customer: CustomerInput
    interventions: WhatIfInterventions
