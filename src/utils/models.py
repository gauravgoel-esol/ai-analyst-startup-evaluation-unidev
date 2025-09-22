from pydantic import BaseModel
from typing import Dict, Any

class BenchmarkRequest(BaseModel):
    document_text: str
    sector: str

class RiskRequest(BaseModel):
    document_text: str

class GrowthRequest(BaseModel):
    document_text: str
    investor_weights: dict

class StartupEvaluationRequest(BaseModel):
    startup_data: Dict[str, Any]
    weights: Dict[str, float]

class HybridStartupRequest(BaseModel):
    structured_data: Dict[str, Any]
    unstructured_text: str
    weights: Dict[str, float]