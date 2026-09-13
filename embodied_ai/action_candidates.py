"""Public-observation action proposals shared by lightweight policies."""
from __future__ import annotations

from typing import Any

from .schemas import ActionRequest


_BASIC_ACTIONS = (
    ActionRequest(type="move", direction="north"),
    ActionRequest(type="move", direction="south"),
    ActionRequest(type="move", direction="east"),
    ActionRequest(type="move", direction="west"),
    ActionRequest(type="inspect"),
    ActionRequest(type="wait"),
)


def public_action_candidates(observation: dict[str, Any]) -> list[ActionRequest]:
    """Return deterministic, locally plausible actions without hidden state.

    These are candidates only; the Rust authority remains responsible for final
    validation because an observed entity may change before an action arrives.
    """
    allowed = set(observation.get("allowed_action_types", []))
    candidates = [action for action in _BASIC_ACTIONS if action.type in allowed]
    entities = sorted(
        (
            (entity, cell.get("relative_position", {}))
            for cell in observation.get("visible_cells", [])
            if isinstance(cell, dict)
            for entity in cell.get("entities", [])
            if isinstance(entity, dict)
        ),
        key=lambda item: (str(item[0].get("type", "")), str(item[0].get("id", ""))),
    )
    for entity, cell_position in entities:
        entity_id = entity.get("id")
        position = entity.get("relative_position", cell_position)
        if not isinstance(entity_id, str) or not entity_id or not isinstance(position, dict):
            continue
        distance = abs(position.get("x", 99)) + abs(position.get("y", 99))
        if entity.get("type") in {"door", "container"} and distance <= 1 and "open" in allowed:
            candidates.append(ActionRequest(type="open", target_id=entity_id))
        elif entity.get("type") == "item" and distance == 0 and "pickup" in allowed:
            candidates.append(ActionRequest(type="pickup", item_id=entity_id))
    if "use_item" in allowed:
        for item_id in sorted(str(item) for item in observation.get("agent", {}).get("inventory", [])):
            candidates.append(ActionRequest(type="use_item", item_id=item_id))
    return candidates
