import json
import pytest
from pathlib import Path
from embodied_ai.benchmark import benchmark
from embodied_ai.context import AgentContext, SYSTEM_PROMPT
from embodied_ai.datasets import export_jsonl, export_parquet
from embodied_ai.providers import CautiousProvider, ExplorerProvider, RandomValidProvider
from embodied_ai.schemas import AgentDecision

def test_schema_rejects_unknown_action_fields():
    assert AgentDecision.model_validate({"action":{"type":"move","direction":"east"}}).action.direction == "east"
    try: AgentDecision.model_validate({"action":{"type":"teleport"}})
    except Exception: pass
    else: raise AssertionError("invalid action passed schema validation")

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

def test_exports_and_benchmark(tmp_path: Path):
    jsonl=export_jsonl([{"run_id":"r","step":1}],tmp_path/"events.jsonl")
    parquet=export_parquet([{"run_id":"r","step":1}],tmp_path/"events.parquet")
    assert json.loads(jsonl.read_text())['run_id']=="r" and parquet.exists()
    report=benchmark(2, 10, tmp_path/"bench")
    assert report['runs']==2 and (tmp_path/"bench"/"runs.parquet").exists()

def test_random_provider_returns_allowed_action():
    action=RandomValidProvider(7).choose_action({})
    assert action.type=="move" and action.direction in {"north","south","east","west"}

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
