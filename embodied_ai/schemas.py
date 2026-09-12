"""Canonical versioned wire schemas used by providers, exports, and the UI boundary."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = 1
DATASET_SCHEMA_VERSION = 1

class Position(BaseModel): x: int; y: int
class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["move", "inspect", "pickup", "drop", "use_item", "open", "close", "talk", "give", "wait", "rest"]
    direction: Literal["north", "south", "east", "west"] | None = None
    target_id: str | None = Field(default=None, max_length=128)
    item_id: str | None = Field(default=None, max_length=128)
    message: str | None = Field(default=None, max_length=500)

    @model_validator(mode="before")
    @classmethod
    def validate_action_shape(cls, value):
        if not isinstance(value, dict): return value
        action_type = value.get("type")
        allowed = {
            "move": {"type", "direction"},
            "inspect": {"type", "target_id"},
            "pickup": {"type", "item_id"},
            "drop": {"type", "item_id"},
            "use_item": {"type", "item_id"},
            "open": {"type", "target_id"},
            "close": {"type", "target_id"},
            "talk": {"type", "target_id", "message"},
            "give": {"type", "target_id", "item_id"},
            "wait": {"type"},
            "rest": {"type"},
        }
        required = {"move": "direction", "drop": "item_id", "use_item": "item_id", "open": "target_id", "close": "target_id", "talk": "target_id", "give": "target_id"}
        if action_type not in allowed: return value
        provided = {field for field, field_value in value.items() if field_value is not None}
        unexpected = provided - allowed[action_type]
        if unexpected: raise ValueError(f"fields are not valid for {action_type}: {sorted(unexpected)}")
        if required.get(action_type) and not value.get(required[action_type]):
            raise ValueError(f"{required[action_type]} is required for {action_type}")
        if action_type == "give" and not value.get("item_id"):
            raise ValueError("item_id is required for give")
        return value

class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ActionRequest
    decision_summary: str = Field(default="", max_length=500)

class RunMetadata(BaseModel):
    protocol_version: int = PROTOCOL_VERSION
    dataset_schema_version: int = DATASET_SCHEMA_VERSION
    run_id: str
    scenario_id: str
    scenario_version: int
    seed: int
    provider: str
    model: str | None = None

class BenchmarkResult(BaseModel):
    scenario_id: str
    provider: str
    runs: int
    success_rate: float
    mean_score: float
    output_directory: str
