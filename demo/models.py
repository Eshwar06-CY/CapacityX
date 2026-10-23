"""Pydantic models for CapacityX API endpoints in demo/server.py."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ShipmentIncidentRequest(BaseModel):
    """Logistics incident payload submitted for decision evaluation."""

    shipment_id: str = Field(..., description="Unique identifier for the shipment (e.g. SHP-7821-X)")
    delivery_deadline: str = Field(..., description="Target delivery deadline (ISO 8601 string or human readable)")
    truck_breakdown: bool = Field(..., description="Whether the primary transport vehicle experienced a breakdown")
    breakdown_severity: Optional[str] = Field(default="MODERATE", description="Severity (LOW, MODERATE, CRITICAL)")
    replacement_vehicle_available: bool = Field(..., description="Whether a backup vehicle is currently nearby")
    customer_notified: bool = Field(..., description="Whether customer has been notified")
    cargo_type: Optional[str] = Field(default="Standard Freight", description="Type of cargo")
    origin: Optional[str] = Field(default="Distribution Hub Alpha", description="Origin depot")
    destination: Optional[str] = Field(default="Regional Retail Store #412", description="Destination drop-off")
    estimated_distance_remaining_km: Optional[float] = Field(default=65.0, description="Remaining distance in km")


class SimulatedActionItem(BaseModel):
    """An individual action item within the execution preview."""
    system: str
    action: str
    status: str
    is_live_action: bool = False


class SimulatedOutcome(BaseModel):
    """Structured decision outcome with typed tool selection and confidence metrics."""
    selected_action: str = Field(..., description="Selected tool among 5 canonical tools")
    action_type: str = Field(..., description="Normalized uppercase action code")
    model_confidence: Optional[float] = Field(default=None, description="Calibrated confidence score (0.0 to 1.0)")
    alternatives: Dict[str, float] = Field(default_factory=dict, description="Probabilities across alternatives")
    human_review_required: bool = Field(default=False, description="Flag indicating human review is mandatory")
    human_review_reason: Optional[str] = Field(default=None, description="Detailed explanation if review is required")
    engine: str = Field(default="SAFE_FALLBACK", description="Engine identifier")
    is_real_model: bool = Field(default=False, description="Whether output came from live model inference")
    fallback_reason: Optional[str] = Field(default=None, description="Explanation if fallback applied")
    model_id: Optional[str] = Field(default="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610")
    urgency: str
    headline: str
    rationale: str
    actions_preview: List[SimulatedActionItem]
    estimated_delay_minutes: int
    deadline_met_projected: bool
    sla_risk_level: str
    customer_status_recommended: str
    bedrock_validation: Optional[Dict[str, Any]] = None


class DecisionResponse(BaseModel):
    """Envelope for API decision responses."""
    is_simulated: bool
    real_action_taken: bool = False
    environment: str
    disclaimer: str
    shipment_id: str
    delivery_deadline: str
    cargo_type: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    timestamp: str
    outcome: SimulatedOutcome
