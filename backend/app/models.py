"""Data models for CapacityX Logistics Decision API.

These models define the contract for shipment incident triage and tool-selection decisions,
supporting both real Strands Decider 2B inference and safe deterministic fallbacks.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ShipmentIncidentRequest(BaseModel):
    """Logistics incident payload submitted for decision evaluation."""

    shipment_id: str = Field(
        ...,
        description="Unique identifier for the shipment (e.g. SHP-7821-X)",
        example="SHP-7821-X",
    )
    delivery_deadline: str = Field(
        ...,
        description="Target delivery deadline (ISO 8601 string or human readable)",
        example="2026-10-20T16:30:00Z",
    )
    truck_breakdown: bool = Field(
        ...,
        description="Whether the primary transport vehicle experienced a mechanical or operational breakdown",
        example=True,
    )
    breakdown_severity: Optional[str] = Field(
        default="MODERATE",
        description="Severity of breakdown if applicable (LOW, MODERATE, CRITICAL)",
        example="CRITICAL",
    )
    replacement_vehicle_available: bool = Field(
        ...,
        description="Whether a secondary or backup transport vehicle is currently available nearby",
        example=True,
    )
    customer_notified: bool = Field(
        ...,
        description="Whether the receiving customer has already been notified of disruption/status",
        example=False,
    )
    cargo_type: Optional[str] = Field(
        default="Standard Freight",
        description="Type of cargo (e.g., Perishable/Cold-chain, Pharmaceuticals, Standard Freight, Hazmat)",
        example="Cold-chain Perishables",
    )
    origin: Optional[str] = Field(
        default="Distribution Hub Alpha",
        description="Origin depot or distribution center",
        example="Chicago Depot North",
    )
    destination: Optional[str] = Field(
        default="Regional Retail Store #412",
        description="Destination drop-off location",
        example="Milwaukee Logistics Park",
    )
    estimated_distance_remaining_km: Optional[float] = Field(
        default=65.0,
        description="Estimated remaining distance to destination in kilometers",
        example=65.0,
    )


class SimulatedActionItem(BaseModel):
    """An individual action item within the execution preview."""
    system: str
    action: str
    status: str
    is_live_action: bool = False


class SimulatedOutcome(BaseModel):
    """Structured decision outcome with typed tool selection and confidence metrics."""
    # Typed decision tool selection among the 5 tools
    selected_action: str = Field(
        ...,
        description="Selected tool: dispatch_replacement, reroute_shipment, crossdock, escalate_to_human, continue_monitoring",
    )
    action_type: str = Field(
        ...,
        description="Normalized uppercase action code for dashboard compatibility",
    )
    model_confidence: Optional[float] = Field(
        default=None,
        description="Calibrated probability score (0.0 to 1.0) from Strands Decider 2B",
    )
    alternatives: Dict[str, float] = Field(
        default_factory=dict,
        description="Probabilistic distribution across alternative decision options",
    )
    human_review_required: bool = Field(
        default=False,
        description="Flag indicating human dispatcher confirmation is mandatory",
    )
    engine: str = Field(
        default="SAFE_FALLBACK",
        description="Engine identifier: STRANDS_DECIDER_2B_REAL, AMAZON_BEDROCK_NOVA_MICRO, or SAFE_FALLBACK",
    )
    is_real_model: bool = Field(
        default=False,
        description="Whether this outcome was produced by live model inference (Strands Decider 2B or Amazon Bedrock Nova Micro)",
    )
    fallback_reason: Optional[str] = Field(
        default=None,
        description="Explanation if safe fallback was applied instead of real model",
    )
    model_id: Optional[str] = Field(
        default="StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
        description="Model checkpoint identifier",
    )

    # Standard metrics and dashboard descriptors
    urgency: str
    headline: str
    rationale: List[str]
    actions_preview: List[SimulatedActionItem]
    estimated_delay_minutes: int
    deadline_met_projected: bool
    sla_risk_level: str
    customer_status_recommended: str
    bedrock_validation: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional complementary operational reasoning from Amazon Bedrock Nova Micro.",
    )


class DecisionResponse(BaseModel):
    """Response envelope returned by decision endpoints."""
    is_simulated: bool = True
    real_action_taken: bool = False
    environment: str = "LOCAL_MOCK_DEV"
    disclaimer: str = (
        "DECISION TRANSPARENCY: Check the 'outcome.engine' and 'outcome.is_real_model' fields "
        "to confirm if this decision originated from the live Strands Decider 2B model or the deterministic safe fallback."
    )
    shipment_id: str
    timestamp: str
    outcome: SimulatedOutcome
