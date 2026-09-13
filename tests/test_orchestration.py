import json
import pytest
from pathlib import Path
from embodied_ai.context import AgentContext, SYSTEM_PROMPT
from embodied_ai.datasets import export_jsonl, export_parquet
from embodied_ai.providers import CallablePolicy, CautiousProvider, ExplorerProvider, RandomValidProvider
from embodied_ai.schemas import AgentDecision

def test_schema_rejects_unknown_action_fields():
    assert AgentDecision.model_validate({"action":{"type":"move","direction":"east"}}).action.direction == "east"
    try: AgentDecision.model_validate({"action":{"type":"teleport"}})
    except Exception: pass
    else: raise AssertionError("invalid action passed schema validation")


def test_agent_metadata_is_bounded_operational_telemetry():
    decision = AgentDecision.model_validate({
        "action": {"type": "wait"},
        "agent_metadata": {"confidence": .8, "value_estimate": -1.5, "policy_entropy": .2, "planner": {"name": "astar", "expanded_nodes": 12, "planning_time_ms": 3.5}},
    })
    assert decision.agent_metadata.planner.name == "astar"
    with pytest.raises(Exception):
        AgentDecision.model_validate({"action": {"type": "wait"}, "agent_metadata": {"confidence": 1.1}})
    with pytest.raises(Exception):
        AgentDecision.model_validate({"action": {"type": "wait"}, "agent_metadata": {"planner": {"name": "", "reasoning": "hidden trace"}}})

def test_context_does_not_contain_world_snapshot():
    context=AgentContext(); context.record({"visible_cells":[]},{"type":"wait"},[])
    assert "WorldSnapshot" not in SYSTEM_PROMPT and "visible_cells" not in context.payload()

@pytest.mark.parametrize("window", [1, 3, 100])
def test_context_window_retains_only_recent_actions_and_observations(window):
    context = AgentContext(memory_window=window)
    for step in range(105):
        context.record({"step": step}, {"type": "inspect", "target_id": str(step)}, [])
    assert len(context.observations) == window
    assert [a["target_id"] for a in context.payload()["recent_actions"]] == [str(i) for i in range(105-window, 105)]

@pytest.mark.parametrize("window", [0, -1, 101, True, 1.5])
def test_context_window_rejects_invalid_bounds(window):
    with pytest.raises(ValueError):
        AgentContext(memory_window=window)

def test_exports_are_written_without_a_second_python_simulator(tmp_path: Path):
    jsonl=export_jsonl([{"run_id":"r","step":1}],tmp_path/"events.jsonl")
    parquet=export_parquet([{"run_id":"r","step":1}],tmp_path/"events.parquet")
    assert json.loads(jsonl.read_text())['run_id']=="r" and parquet.exists()

def test_random_provider_returns_allowed_action():
    action=RandomValidProvider(7).choose_action({})
    assert action.type=="move" and action.direction in {"north","south","east","west"}


def test_callable_policy_keeps_custom_agents_on_the_typed_public_action_boundary():
    resets = []
    policy = CallablePolicy("custom", lambda observation: {"type": "wait"} if observation == {"public": True} else {}, resets.append)
    policy.reset(7)
    assert policy.act({"public": True}).model_dump(exclude_none=True) == {"type": "wait"}
    assert resets == [7]
    try:
        policy.act({"public": False})
    except Exception:
        pass
    else:
        raise AssertionError("custom policies bypassed typed action validation")

def test_baseline_policy_lifecycle_resets_deterministically():
    explorer=ExplorerProvider()
    assert explorer.act({}).type == "inspect"
    assert explorer.act({}).type == "move"
    explorer.reset()
    assert explorer.act({}).type == "inspect"
    random=RandomValidProvider(9)
    first=random.act({})
    random.reset(9)
    assert random.act({}) == first

def test_lightweight_policy_baselines_are_deterministic_and_hazard_aware():
    explorer=ExplorerProvider()
    assert [explorer.choose_action({}).type for _ in range(5)]==["inspect","move","move","move","move"]
    observation={"visible_cells":[
        {"relative_position":{"x":1,"y":0},"terrain":"floor","entities":[{"type":"hazard","state":"active"}]},
        {"relative_position":{"x":0,"y":1},"terrain":"floor","entities":[]},
    ]}
    cautious=CautiousProvider()
    assert cautious.choose_action(observation).type=="inspect"
    safe=cautious.choose_action(observation)
    assert safe.type=="move" and safe.direction=="south"

def test_give_action_requires_both_target_and_item():
    assert AgentDecision.model_validate({"action":{"type":"give","target_id":"npc","item_id":"water"}}).action.item_id=="water"
    with pytest.raises(Exception): AgentDecision.model_validate({"action":{"type":"give","target_id":"npc"}})
