from embodied_ai.schemas import ActionRequest
from embodied_ai.schemas import AgentDecision
from embodied_ai.runner import RustRunClient
from embodied_ai import runner
from embodied_ai.engine import Action
from pydantic import ValidationError
import pytest
import json
from pathlib import Path

def test_action_wire_schema_omits_none_fields():
    assert ActionRequest(type="move",direction="east").model_dump(exclude_none=True)=={"type":"move","direction":"east"}


def test_provider_latency_is_encoded_as_integer_for_rust_wire_contract():
    captured = []
    class Response:
        def raise_for_status(self): return self
        def json(self): return {}
    class Client:
        def post(self, path, json):
            captured.append(json)
            return Response()
    client = object.__new__(RustRunClient)
    client.client = Client()
    client.record_decision("r", ActionRequest(type="wait"), "Wait.", "gemini", "fake", 12.7, None)
    assert captured[0]["latency_ms"] == 13
    assert type(captured[0]["latency_ms"]) is int


def test_committed_action_schema_matches_the_canonical_python_wire_model():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "action-request.v1.json").read_text())
    generated = ActionRequest.model_json_schema()
    assert schema["additionalProperties"] == generated["additionalProperties"]
    assert schema["required"] == generated["required"]
    assert schema["properties"]["type"]["enum"] == generated["properties"]["type"]["enum"]
    for field, limit in (("target_id", 128), ("item_id", 128), ("message", 500)):
        assert schema["properties"][field]["anyOf"][0]["maxLength"] == limit


def test_action_schema_rejects_invalid_field_combinations():
    for payload in ({"type": "wait", "direction": "north"}, {"type": "move"}, {"type": "open", "item_id": "key"}):
        try: ActionRequest(**payload)
        except ValidationError: pass
        else: raise AssertionError(f"invalid action shape was accepted: {payload}")
    assert ActionRequest(type="move", direction="north", target_id=None, item_id=None).direction == "north"


def test_authoritative_run_creation_sends_an_optional_max_step_override():
    calls = []
    class Response:
        def raise_for_status(self): return self
        def json(self): return {"run_id": "test"}
    class Client:
        def post(self, path, json):
            calls.append((path, json)); return Response()
    client = object.__new__(RustRunClient)
    client.client = Client()
    assert client.create(42, max_steps=7, observation_mode="rich") == {"run_id": "test"}
    assert calls == [("/api/runs", {"seed": 42, "max_steps": 7, "observation_mode": "rich"})]


@pytest.mark.parametrize("failure", [RuntimeError("ProviderUnavailable"), ValueError("Malformed provider response")])
def test_provider_failure_marks_the_authoritative_run_as_provider_error(monkeypatch, failure):
    calls = []
    class Client:
        def __init__(self, _base_url): pass
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): return {"run_id": "r", "observation": {"allowed_action_types": []}}
        def status(self, _run_id): return {"done": False, "paused": False, "step": 0}
        def provider_error(self, run_id): calls.append(run_id); return {"terminal_reason": "provider_error"}
        def close(self): pass
    class Provider:
        name = "failing"
        def choose_action(self, _observation): raise failure
    monkeypatch.setattr(runner, "RustRunClient", Client)
    result = runner.run_remote(Provider(), 7)
    assert result.terminal_reason == "provider_error" and result.records == [] and calls == ["r"]


def test_trajectory_records_the_exact_compact_provider_context(monkeypatch):
    class Client:
        def __init__(self, _base_url): self.steps = 0
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): return {"run_id": "r", "observation": {"allowed_action_types": ["wait"]}}
        def status(self, _run_id): return {"done": False, "paused": False, "step": self.steps}
        def record_decision(self, _run_id, _action, _summary, _provider, _model, _latency, _token_usage): return {}
        def step(self, _run_id, _action):
            self.steps += 1
            return {"step_number": self.steps, "observation": {"allowed_action_types": ["wait"]}, "events": [{"type": "NpcSpoke", "message": "The exit key is in the locker."}], "reward": -1, "done": self.steps == 2, "terminal_reason": "escaped" if self.steps == 2 else None, "metrics": {}}
        def close(self): pass
    class Provider:
        name = "context-test"
        async def choose(self, _observation, _context):
            return AgentDecision(action=ActionRequest(type="wait"), decision_summary="Safe progress."), 3
    monkeypatch.setattr(runner, "RustRunClient", Client)
    result = runner.run_remote(Provider(), 7)
    assert result.records[0]["agent_context"] == {"recent_actions": [], "known_facts": []}
    assert result.records[1]["agent_context"] == {"recent_actions": [{"type": "wait"}], "known_facts": ["The exit key is in the locker."]}
    assert result.records[0]["step_latency_ms"] >= 0
    assert result.records[1]["control_elapsed_ms"] >= result.records[0]["control_elapsed_ms"]


def test_runner_exits_without_a_provider_call_when_researcher_aborts(monkeypatch):
    class Client:
        def __init__(self, _base_url): pass
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): return {"run_id": "r", "observation": {"allowed_action_types": ["wait"]}}
        def status(self, _run_id): return {"done": True, "paused": False, "step": 4, "terminal_reason": "aborted"}
        def close(self): pass
    class Provider:
        name = "must-not-run"
        def choose_action(self, _observation): raise AssertionError("provider should not be called after abort")
    monkeypatch.setattr(runner, "RustRunClient", Client)
    result = runner.run_remote(Provider(), 7)
    assert result.terminal_reason == "aborted" and result.steps == 4 and result.records == []


@pytest.mark.parametrize("budget", ["tokens", "wall"])
def test_runner_stops_before_submitting_an_action_that_exceeds_token_budget(monkeypatch, budget):
    calls = []
    class Client:
        def __init__(self, _base_url): pass
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): return {"run_id": "r", "observation": {"allowed_action_types": ["wait"]}}
        def status(self, _run_id): return {"done": False, "paused": False, "step": 0}
        def stop(self, run_id, reason): calls.append((run_id, reason)); return {}
        def provider_error(self, _run_id): raise AssertionError("budget stop is not a provider error")
        def close(self): pass
    class Provider:
        name = "metered"
        async def choose(self, _observation, _context):
            self.last_token_usage = {"total_tokens": 11}
            return AgentDecision(action=ActionRequest(type="wait"), decision_summary="Safe."), 1
    monkeypatch.setattr(runner, "RustRunClient", Client)
    if budget == "wall":
        from types import SimpleNamespace
        ticks = iter([0.0, 0.1, 2.0])
        monkeypatch.setattr(runner, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
        result = runner.run_remote(Provider(), 7, max_wall_seconds=1)
        reason = "client_timeout"
    else:
        result = runner.run_remote(Provider(), 7, max_total_tokens=10)
        reason = "token_budget_exhausted"
    assert result.terminal_reason == reason
    assert result.steps == 0 and result.records == []
    assert calls == [("r", reason)]


def test_runner_can_continue_an_existing_nonterminal_authoritative_run(monkeypatch):
    calls = []
    class Client:
        def __init__(self, _base_url): self.finished = False
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): raise AssertionError("resume must not create a new run")
        def status(self, _run_id): return {"done": self.finished, "paused": False, "step": 4, "terminal_reason": "escaped" if self.finished else None}
        def observation(self, run_id): calls.append(("observation", run_id)); return {"allowed_action_types": ["wait"]}
        def record_decision(self, *_args): pass
        def step(self, run_id, _action):
            self.finished = True; calls.append(("step", run_id))
            return {"step_number": 5, "observation": {"allowed_action_types": ["wait"]}, "events": [], "reward": 100, "done": True, "terminal_reason": "escaped", "metrics": {}}
        def provider_error(self, _run_id): raise AssertionError
        def close(self): pass
    class Provider:
        name = "resume"
        def choose_action(self, _observation): return Action(type="wait")
    monkeypatch.setattr(runner, "RustRunClient", Client)
    result = runner.run_remote(Provider(), 7, resume_run_id="existing")
    assert result.run_id == "existing" and result.steps == 5 and result.terminal_reason == "escaped"
    assert calls == [("observation", "existing"), ("step", "existing")]


def test_runner_stops_when_wall_clock_budget_is_already_exhausted(monkeypatch):
    calls = []
    class Client:
        def __init__(self, _base_url): pass
        def scenarios(self): return [{"id": "survival_room", "version": 3}]
        def create(self, *_args): return {"run_id": "r", "observation": {"allowed_action_types": ["wait"]}}
        def stop(self, run_id, reason): calls.append((run_id, reason)); return {}
        def close(self): pass
    class Provider:
        name = "must-not-run"
        def choose_action(self, _observation): raise AssertionError("time budget should stop first")
    ticks = iter([0.0, 2.0])
    monkeypatch.setattr(runner, "RustRunClient", Client)
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(ticks))
    result = runner.run_remote(Provider(), 7, max_wall_seconds=1)
    assert result.terminal_reason == "client_timeout" and calls == [("r", "client_timeout")]


def test_runner_restores_a_persisted_replay_before_provider_control(monkeypatch):
    class Client:
        def __init__(self, _base_url): self.finished=False
        def scenarios(self): return [{"id": "survival_room", "version": 4}]
        def restore(self, replay_id):
            assert replay_id == "saved"
            return {"run_id":"rebuilt","snapshot":{"step":3},"observation":{"allowed_action_types":["wait"]}}
        def status(self, _run_id): return {"done":self.finished,"paused":False,"step":3}
        def record_decision(self, *_args): pass
        def step(self, _run_id, _action):
            self.finished=True
            return {"step_number":4,"observation":{"allowed_action_types":["wait"]},"events":[],"reward":100,"done":True,"terminal_reason":"escaped","metrics":{}}
        def provider_error(self, _run_id): raise AssertionError
        def close(self): pass
    class Provider:
        name="restored"
        def choose_action(self, _observation): return Action(type="wait")
    monkeypatch.setattr(runner,"RustRunClient",Client)
    result=runner.run_remote(Provider(),7,restore_replay_id="saved")
    assert result.run_id=="rebuilt" and result.steps==4 and result.terminal_reason=="escaped"
