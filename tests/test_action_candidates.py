from embodied_ai.action_candidates import public_action_candidates


def test_public_action_candidates_filter_by_local_range_and_allowed_types():
    observation = {
        "agent": {"inventory": ["water"]},
        "allowed_action_types": ["move", "open", "pickup", "use_item"],
        "visible_cells": [
            {"relative_position": {"x": 0, "y": 0}, "entities": [{"id": "key", "type": "item"}]},
            {"relative_position": {"x": 1, "y": 0}, "entities": [{"id": "door", "type": "door"}]},
            {"relative_position": {"x": 2, "y": 0}, "entities": [{"id": "far", "type": "item"}]},
        ],
    }
    payloads = [action.model_dump(exclude_none=True) for action in public_action_candidates(observation)]
    assert {"type": "pickup", "item_id": "key"} in payloads
    assert {"type": "open", "target_id": "door"} in payloads
    assert {"type": "use_item", "item_id": "water"} in payloads
    assert all("far" not in str(payload) for payload in payloads)
    assert {"type": "inspect"} not in payloads
