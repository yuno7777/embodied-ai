"""Quarantined legacy Python simulator.

Rust ``sim-core`` is the only supported simulation authority. This module remains
temporarily for historical migration coverage and must not be imported by new
production code.
"""
from __future__ import annotations

import json, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = ROOT / "scenarios" / "survival_room" / "scenario.json"
DIRECTIONS = {"north": (0,-1), "south": (0,1), "west": (-1,0), "east": (1,0)}

@dataclass
class Action:
    type: str
    direction: str | None = None
    target_id: str | None = None
    item_id: str | None = None

@dataclass
class Agent:
    position: tuple[int,int]
    health: int = 100
    energy: int = 100
    hydration: int = 100
    inventory: list[str] = field(default_factory=list)

class Scenario:
    def __init__(self, raw: dict[str, Any]): self.raw = raw
    @classmethod
    def load(cls, path: Path = SCENARIO_PATH): return cls(json.loads(path.read_text()))

class Environment:
    """Legacy-only simulator retained during the Rust-authority migration."""
    protocol_version = 1
    def __init__(self, scenario: Scenario | None = None, seed: int = 42):
        self.scenario = scenario or Scenario.load(); self.seed = seed; self.run_id = str(uuid.uuid4())
        self.agent = Agent(tuple(self.scenario.raw["spawn"])); self.step_number = 0; self.events: list[dict] = []
        self.done = False; self.terminal_reason: str | None = None; self.invalid_actions = 0; self.hazard_damage = 0; self.visited = {self.agent.position}
    def _event(self, kind: str, message: str):
        event={"event_id":str(uuid.uuid4()),"type":kind,"step":self.step_number,"message":message}; self.events.append(event); return event
    def _at(self, key: str, position: tuple[int,int]):
        return next((x for x in self.scenario.raw.get(key,[]) if tuple(x["position"]) == position), None)
    def observe(self) -> dict:
        radius=self.scenario.raw["vision_radius"]; x,y=self.agent.position; cells=[]
        for yy in range(y-radius,y+radius+1):
          for xx in range(x-radius,x+radius+1):
            if abs(xx-x)+abs(yy-y)>radius: continue
            p=(xx,yy); terrain="wall" if list(p) in self.scenario.raw["walls"] else "floor"; entities=[]
            if (d:=self._at("doors",p)): entities.append({"id":d["id"],"type":"door","state":"open" if d.get("open") else ("locked" if d["locked"] else "closed")})
            if (i:=self._at("items",p)): entities.append({"id":i["id"],"type":"item","name":i["name"]})
            if (h:=self._at("hazards",p)): entities.append({"id":h["id"],"type":"hazard"})
            if tuple(self.scenario.raw["npc"]["position"])==p: entities.append({"id":"caretaker","type":"npc"})
            cells.append({"relative_position":{"dx":xx-x,"dy":yy-y},"terrain":terrain,"entities":entities})
        return {"protocol_version":1,"run_id":self.run_id,"step":self.step_number,"agent":{"health":self.agent.health,"energy":self.agent.energy,"hydration":self.agent.hydration,"inventory":self.agent.inventory},"goal":self.scenario.raw["goal"],"visible_cells":cells,"recent_events":[e["message"] for e in self.events[-5:]],"allowed_action_types":["move","inspect","pickup","use_item","open","talk","wait","rest"]}
    def snapshot(self):
        return {"run_id":self.run_id,"step":self.step_number,"agent":asdict(self.agent),"items":self.scenario.raw["items"],"doors":self.scenario.raw["doors"],"hazards":self.scenario.raw["hazards"],"npc":self.scenario.raw["npc"],"done":self.done,"terminal_reason":self.terminal_reason}
    def step(self, action: Action) -> dict:
        if self.done: return {"observation":self.observe(),"reward":0,"done":True,"terminal_reason":self.terminal_reason,"events":[self._event("RunAlreadyTerminal","Run is terminal")]}
        self.step_number += 1; emitted=[]
        def emit(k,m): emitted.append(self._event(k,m))
        if action.type == "move" and action.direction in DIRECTIONS:
            dx,dy=DIRECTIONS[action.direction]; target=(self.agent.position[0]+dx,self.agent.position[1]+dy)
            door=self._at("doors",target)
            # A closed door occupies a boundary wall cell; an open door is passable.
            if (list(target) in self.scenario.raw["walls"] and not (door and door.get("open"))) or (door and not door.get("open")):
                self.invalid_actions+=1; emit("MoveBlocked","Movement blocked.")
            else:
                self.agent.position=target; self.visited.add(target); self.agent.energy=max(0,self.agent.energy-2); self.agent.hydration=max(0,self.agent.hydration-1); emit("AgentMoved",f"Moved {action.direction}.")
        elif action.type == "pickup":
            item=self._at("items",self.agent.position)
            if item and item["id"] not in self.agent.inventory: self.agent.inventory.append(item["id"]); emit("ItemPickedUp",f"Picked up {item['name']}.")
            else: self.invalid_actions+=1; emit("InvalidAction","Nothing to pick up here.")
        elif action.type == "use_item" and action.item_id in self.agent.inventory:
            item=next(i for i in self.scenario.raw["items"] if i["id"]==action.item_id)
            self.agent.energy=min(100,self.agent.energy+item["effect"].get("energy",0)); self.agent.hydration=min(100,self.agent.hydration+item["effect"].get("hydration",0)); self.agent.inventory.remove(action.item_id); emit("ItemConsumed",f"Used {item['name']}.")
        elif action.type == "open" and action.target_id == "exit_door" and self.agent.position == (7,3):
            door=self.scenario.raw["doors"][0]
            if door["key_id"] in self.agent.inventory: door["open"]=True; emit("DoorOpened","The exit door opens.")
            else: self.invalid_actions+=1; emit("DoorLocked","The exit door requires the brass key.")
        elif action.type == "talk" and self.agent.position == tuple(self.scenario.raw["npc"]["position"]): emit("NpcSpoke",self.scenario.raw["npc"]["hint"])
        elif action.type in {"inspect","wait","rest"}: emit("ActionCompleted",f"{action.type.title()} completed.")
        else: self.invalid_actions+=1; emit("InvalidAction","That action is not valid here.")
        hazard=self._at("hazards",self.agent.position)
        if hazard: self.agent.health=max(0,self.agent.health-hazard["damage"]); self.hazard_damage+=hazard["damage"]; emit("HazardTriggered","Electrical sparks cause damage.")
        if self.agent.position == (8,3): self.done=True; self.terminal_reason="escaped"; emit("GoalCompleted","You escaped the survival room.")
        elif self.agent.health<=0: self.done=True; self.terminal_reason="dead"; emit("AgentDied","Health reached zero.")
        elif self.step_number>=self.scenario.raw["max_steps"]: self.done=True; self.terminal_reason="timeout"; emit("RunTimedOut","Maximum steps reached.")
        return {"observation":self.observe(),"reward":100 if self.terminal_reason=="escaped" else (-100 if self.done else -1),"done":self.done,"terminal_reason":self.terminal_reason,"events":emitted,"step_number":self.step_number,"simulation_time":self.step_number}
