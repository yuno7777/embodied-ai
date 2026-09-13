"""Canonical versioned wire schemas used by providers, exports, and the UI boundary."""
from __future__ import annotations
import math
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROTOCOL_VERSION = 1
DATASET_SCHEMA_VERSION = 1

class Position(BaseModel): x: int; y: int


ActionType = Literal["move", "inspect", "pickup", "drop", "use_item", "open", "close", "talk", "give", "wait", "rest"]


class VisibleEntity(BaseModel):
    """One public entity descriptor emitted by the Rust sensor boundary."""
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    type: str = Field(min_length=1, max_length=128)
    state: str | None = None
    name: str | None = None


class VisibleCell(BaseModel):
    model_config = ConfigDict(extra="forbid")
    relative_position: Position
    terrain: Literal["floor", "wall"]
    entities: list[VisibleEntity]


class AgentObservation(BaseModel):
    """Public body state, deliberately excluding identity and absolute position."""
    model_config = ConfigDict(extra="forbid")
    facing: Literal["north", "south", "east", "west"]
    health: int = Field(ge=0, le=100)
    energy: int = Field(ge=0, le=100)
    hydration: int = Field(ge=0, le=100)
    inventory: list[str]
    max_inventory: int = Field(ge=0)
    max_inventory_weight: int = Field(ge=0)
    status_effects: list[str]


class Observation(BaseModel):
    """Canonical v1 policy observation returned by the authoritative Rust API."""
    model_config = ConfigDict(extra="forbid")
    protocol_version: Literal[PROTOCOL_VERSION]
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    observation_mode: Literal["minimal", "normal", "rich", "oracle", "noisy"]
    agent: AgentObservation
    goal: str
    visible_cells: list[VisibleCell]
    recent_events: list[str]
    perception_note: str | None = None
    allowed_action_types: list[ActionType]


class WorldGeneratorConfig(BaseModel):
    """Strict generator configuration copied into every generated world manifest."""
    model_config = ConfigDict(extra="forbid")
    generator_version: Literal[1]
    min_width: int = Field(ge=5)
    max_width: int = Field(ge=5)
    min_height: int = Field(ge=5)
    max_height: int = Field(ge=5)
    max_attempts: int = Field(ge=1)
    min_rooms: int = Field(ge=2, le=3)
    max_rooms: int = Field(ge=2, le=3)
    hazard_kinds: list[Literal["electrical", "fire", "toxic_gas"]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "WorldGeneratorConfig":
        if self.min_width > self.max_width or self.min_height > self.max_height:
            raise ValueError("generator minimum dimensions cannot exceed maximum dimensions")
        if self.min_rooms > self.max_rooms:
            raise ValueError("generator minimum rooms cannot exceed maximum rooms")
        if self.max_rooms == 3 and self.min_width < 7:
            raise ValueError("three-room worlds require a minimum width of 7")
        return self


class WorldValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    geometry_valid: bool
    spawn_valid: bool
    required_key_reachable: bool
    exit_reachable: bool
    solvable: bool


class WorldManifest(BaseModel):
    """Canonical v1 generated-world identity from Rust, with frozen scenario data."""
    model_config = ConfigDict(extra="forbid")
    manifest_version: Literal[1]
    generator_version: Literal[1]
    seed: int = Field(ge=0)
    generator_config: WorldGeneratorConfig
    world_hash: str = Field(pattern=r"^fnv1a64:[0-9a-f]+$")
    dimensions: Position
    generation_attempt: int = Field(ge=0)
    validation: WorldValidation
    scenario: dict[str, object]


class RewardBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid")
    baseline: int
    progress: int
    invalid_action_penalty: int
    hazard_penalty: int
    terminal: int


class Event(BaseModel):
    """Authoritative event emitted by a completed Rust transition."""
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    step: int = Field(ge=0)
    simulation_time: int = Field(ge=0)
    timestamp: str = Field(min_length=1)
    type: str = Field(min_length=1)
    message: str


class Metrics(BaseModel):
    """Evaluator-owned v1 metrics returned with each authoritative step."""
    model_config = ConfigDict(extra="forbid")
    escaped: bool
    alive: bool
    steps_taken: int = Field(ge=0)
    simulated_time: int = Field(ge=0)
    final_health: int = Field(ge=0, le=100)
    final_energy: int = Field(ge=0, le=100)
    final_hydration: int = Field(ge=0, le=100)
    invalid_actions: int = Field(ge=0)
    repeated_invalid_actions: int = Field(ge=0)
    unique_cells_visited: int = Field(ge=0)
    useful_items_acquired: int = Field(ge=0)
    carried_weight: int = Field(ge=0)
    inventory_weight_capacity: int = Field(ge=0)
    exploration_coverage: float = Field(ge=0, le=1)
    action_diversity: int = Field(ge=0)
    repeated_actions: int = Field(ge=0)
    action_repetition_rate: float = Field(ge=0, le=1)
    resource_efficiency: float = Field(ge=0, le=1)
    hazard_damage_taken: int = Field(ge=0)
    npc_interactions: int = Field(ge=0)
    unnecessary_actions: int = Field(ge=0)
    recovery_after_failure: bool
    discovered_doors: int = Field(ge=0)
    discovered_items: int = Field(ge=0)
    discovered_hazards: int = Field(ge=0)
    discovered_npcs: int = Field(ge=0)
    first_discovery_steps: dict[str, int]
    milestones: dict[str, int]
    score_task_success: float
    score_health: float
    score_invalid_action_penalty: float
    score_step_penalty: float
    normalized_score: float = Field(ge=0, le=100)


class ActionResult(BaseModel):
    """Canonical v1 Rust transition result for RL, trajectories, and replays."""
    model_config = ConfigDict(extra="forbid")
    observation: Observation
    reward: int
    reward_breakdown: RewardBreakdown
    done: bool
    terminal_reason: str | None = None
    events: list[Event]
    step_number: int = Field(ge=0)
    simulation_time: int = Field(ge=0)
    metrics: Metrics

    @model_validator(mode="after")
    def validate_terminal_state(self) -> "ActionResult":
        if self.done != (self.terminal_reason is not None):
            raise ValueError("terminal_reason must be present exactly when done")
        return self


class ActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: ActionType
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

class PlannerMetadata(BaseModel):
    """Bounded operational telemetry, not a natural-language reasoning trace."""
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    expanded_nodes: int | None = Field(default=None, ge=0)
    planning_time_ms: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_finite_values(self) -> "PlannerMetadata":
        if self.planning_time_ms is not None and not math.isfinite(self.planning_time_ms):
            raise ValueError("planning_time_ms must be finite")
        return self


class AgentMetadata(BaseModel):
    """Optional auditable policy signals that never affect Rust transitions."""
    model_config = ConfigDict(extra="forbid")
    confidence: float | None = Field(default=None, ge=0, le=1)
    value_estimate: float | None = None
    policy_entropy: float | None = Field(default=None, ge=0)
    planner: PlannerMetadata | None = None

    @model_validator(mode="after")
    def validate_finite_values(self) -> "AgentMetadata":
        if any(value is not None and not math.isfinite(value) for value in (self.confidence, self.value_estimate, self.policy_entropy)):
            raise ValueError("agent metadata values must be finite")
        return self


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: ActionRequest
    decision_summary: str = Field(default="", max_length=500)
    agent_metadata: AgentMetadata | None = None

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
