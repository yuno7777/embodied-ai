//! Deterministic authoritative world model. Providers can propose actions only.
use chrono::{DateTime, Utc};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;
use uuid::Uuid;

pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationMode {
    Minimal,
    #[default]
    Normal,
    Rich,
}

#[derive(Debug, Error)]
pub enum SimError {
    #[error("invalid scenario: {0}")]
    Scenario(String),
    #[error("invalid action: {0}")]
    InvalidAction(String),
    #[error("run is terminal")]
    Terminal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pos {
    pub x: i32,
    pub y: i32,
}
impl Pos {
    fn shifted(self, direction: Direction) -> Self {
        let (x, y) = match direction {
            Direction::North => (0, -1),
            Direction::South => (0, 1),
            Direction::East => (1, 0),
            Direction::West => (-1, 0),
        };
        Self {
            x: self.x + x,
            y: self.y + y,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    North,
    South,
    East,
    West,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Move {
        direction: Direction,
    },
    Inspect {
        target_id: Option<String>,
    },
    Pickup {
        item_id: Option<String>,
    },
    Drop {
        item_id: String,
    },
    UseItem {
        item_id: String,
    },
    Open {
        target_id: String,
    },
    Close {
        target_id: String,
    },
    Talk {
        target_id: String,
        message: Option<String>,
    },
    Give {
        target_id: String,
        item_id: String,
    },
    Wait,
    Rest,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Item {
    pub id: String,
    pub name: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub position: Pos,
    #[serde(default)]
    pub energy: i32,
    #[serde(default)]
    pub hydration: i32,
    #[serde(default)]
    pub health: i32,
    #[serde(default)]
    pub consumable: bool,
    #[serde(default = "default_item_weight")]
    pub weight: u32,
    #[serde(default)]
    pub vision_bonus: i32,
    #[serde(default)]
    pub cures_status_effects: Vec<String>,
}
fn default_item_weight() -> u32 {
    1
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Door {
    pub id: String,
    pub position: Pos,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub open: bool,
    pub key_id: Option<String>,
    #[serde(default)]
    pub is_exit: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hazard {
    pub id: String,
    pub position: Pos,
    #[serde(default = "default_hazard_kind")]
    pub kind: String,
    pub damage: i32,
    #[serde(default)]
    pub energy_drain: i32,
    #[serde(default)]
    pub hydration_drain: i32,
    #[serde(default)]
    pub status_effect: Option<String>,
    #[serde(default = "default_hazard_active")]
    pub active: bool,
}
fn default_hazard_kind() -> String {
    "generic".into()
}
fn default_hazard_active() -> bool {
    true
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Container {
    pub id: String,
    pub position: Pos,
    #[serde(default)]
    pub open: bool,
    #[serde(default)]
    pub locked: bool,
    #[serde(default)]
    pub key_id: Option<String>,
    pub contents: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Npc {
    pub id: String,
    pub position: Pos,
    pub hint: String,
    #[serde(default = "default_npc_disposition")]
    pub disposition: String,
    #[serde(default)]
    pub trust: i32,
    #[serde(default)]
    pub dialogue: Vec<String>,
    #[serde(default)]
    pub patrol: Vec<Pos>,
    #[serde(default)]
    pub inventory: Vec<String>,
}
fn default_npc_disposition() -> String {
    "neutral".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerturbationEffect {
    CloseDoor,
    ActivateHazard,
    DeactivateHazard,
    MoveNpc,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Perturbation {
    pub id: String,
    pub step: u32,
    pub effect: PerturbationEffect,
    pub target_id: String,
    #[serde(default)]
    pub destination: Option<Pos>,
    #[serde(default = "default_perturbation_probability")]
    pub probability_per_mille: u16,
}
fn default_perturbation_probability() -> u16 {
    1000
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub id: String,
    pub name: String,
    pub min: Pos,
    pub max: Pos,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Route {
    pub id: String,
    pub waypoints: Vec<Pos>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub id: String,
    pub name: String,
    pub version: u32,
    pub width: i32,
    pub height: i32,
    pub spawn: Pos,
    pub max_steps: u32,
    #[serde(default)]
    pub time_limit: Option<u32>,
    pub vision_radius: i32,
    pub goal: String,
    #[serde(default)]
    pub walls: Vec<Pos>,
    #[serde(default)]
    pub items: Vec<Item>,
    #[serde(default)]
    pub doors: Vec<Door>,
    #[serde(default)]
    pub hazards: Vec<Hazard>,
    pub npc: Option<Npc>,
    #[serde(default)]
    pub containers: Vec<Container>,
    #[serde(default)]
    pub perturbations: Vec<Perturbation>,
    #[serde(default)]
    pub rooms: Vec<Room>,
    #[serde(default)]
    pub routes: Vec<Route>,
}
impl Scenario {
    pub fn validate(&self) -> Result<(), SimError> {
        if self.width < 3 || self.height < 3 {
            return Err(SimError::Scenario("dimensions must be at least 3".into()));
        }
        if self.max_steps == 0 {
            return Err(SimError::Scenario("max_steps must be positive".into()));
        }
        if self.time_limit == Some(0) {
            return Err(SimError::Scenario(
                "time_limit must be positive when configured".into(),
            ));
        }
        if self.vision_radius < 1 {
            return Err(SimError::Scenario("vision_radius must be positive".into()));
        }
        if !self.in_bounds(self.spawn) {
            return Err(SimError::Scenario("spawn out of bounds".into()));
        }
        if self.walls.iter().any(|wall| !self.in_bounds(*wall)) {
            return Err(SimError::Scenario("wall out of bounds".into()));
        }
        let wall_positions = self.walls.iter().copied().collect::<BTreeSet<_>>();
        if wall_positions.len() != self.walls.len() {
            return Err(SimError::Scenario("wall positions must be unique".into()));
        }
        if wall_positions.contains(&self.spawn) {
            return Err(SimError::Scenario("spawn cannot be inside a wall".into()));
        }
        let room_ids = self
            .rooms
            .iter()
            .map(|room| &room.id)
            .collect::<BTreeSet<_>>();
        if room_ids.len() != self.rooms.len()
            || self.rooms.iter().any(|room| {
                room.id.is_empty()
                    || room.name.is_empty()
                    || !self.in_bounds(room.min)
                    || !self.in_bounds(room.max)
                    || room.min.x > room.max.x
                    || room.min.y > room.max.y
            })
        {
            return Err(SimError::Scenario(
                "rooms require unique ids and valid in-bounds rectangles".into(),
            ));
        }
        let route_ids = self
            .routes
            .iter()
            .map(|route| &route.id)
            .collect::<BTreeSet<_>>();
        if route_ids.len() != self.routes.len()
            || self.routes.iter().any(|route| {
                route.id.is_empty()
                    || route.waypoints.is_empty()
                    || route.waypoints.iter().any(|position| {
                        !self.in_bounds(*position)
                            || self.walls.contains(position)
                                && !self.doors.iter().any(|door| door.position == *position)
                    })
            })
        {
            return Err(SimError::Scenario(
                "routes require unique ids and walkable in-bounds waypoints".into(),
            ));
        }
        if self.items.iter().any(|item| !self.in_bounds(item.position)) {
            return Err(SimError::Scenario("item out of bounds".into()));
        }
        if self.doors.iter().any(|door| !self.in_bounds(door.position)) {
            return Err(SimError::Scenario("door out of bounds".into()));
        }
        if !self.doors.iter().any(|door| door.is_exit) {
            return Err(SimError::Scenario(
                "scenario requires at least one exit door".into(),
            ));
        }
        if self
            .hazards
            .iter()
            .any(|hazard| !self.in_bounds(hazard.position))
        {
            return Err(SimError::Scenario("hazard out of bounds".into()));
        }
        if self
            .containers
            .iter()
            .any(|container| !self.in_bounds(container.position))
        {
            return Err(SimError::Scenario("container out of bounds".into()));
        }
        if self
            .npc
            .as_ref()
            .is_some_and(|npc| !self.in_bounds(npc.position))
        {
            return Err(SimError::Scenario("npc out of bounds".into()));
        }
        if self.npc.as_ref().is_some_and(|npc| {
            npc.patrol
                .iter()
                .any(|position| !self.in_bounds(*position) || self.walls.contains(position))
        }) {
            return Err(SimError::Scenario(
                "npc patrol positions must be in bounds and walkable".into(),
            ));
        }
        let mut ids = BTreeSet::new();
        for id in self
            .items
            .iter()
            .map(|item| &item.id)
            .chain(self.doors.iter().map(|door| &door.id))
            .chain(self.hazards.iter().map(|hazard| &hazard.id))
            .chain(self.containers.iter().map(|container| &container.id))
            .chain(self.npc.iter().map(|npc| &npc.id))
        {
            if id.is_empty() || !ids.insert(id) {
                return Err(SimError::Scenario(format!(
                    "entity id must be unique and non-empty: {id}"
                )));
            }
        }
        let item_ids = self
            .items
            .iter()
            .map(|item| item.id.as_str())
            .collect::<BTreeSet<_>>();
        if self
            .doors
            .iter()
            .filter_map(|door| door.key_id.as_deref())
            .any(|key_id| !item_ids.contains(key_id))
        {
            return Err(SimError::Scenario("door references an unknown key".into()));
        }
        if self.npc.as_ref().is_some_and(|npc| {
            npc.inventory
                .iter()
                .any(|item_id| !item_ids.contains(item_id.as_str()))
        }) {
            return Err(SimError::Scenario("npc references an unknown item".into()));
        }
        if self
            .containers
            .iter()
            .flat_map(|container| &container.contents)
            .any(|item_id| !item_ids.contains(item_id.as_str()))
        {
            return Err(SimError::Scenario(
                "container references an unknown item".into(),
            ));
        }
        if self
            .containers
            .iter()
            .filter_map(|container| container.key_id.as_deref())
            .any(|key_id| !item_ids.contains(key_id))
        {
            return Err(SimError::Scenario(
                "container references an unknown key".into(),
            ));
        }
        if self
            .containers
            .iter()
            .any(|container| container.locked && container.key_id.is_none())
        {
            return Err(SimError::Scenario(
                "locked container must reference a key".into(),
            ));
        }
        for perturbation in &self.perturbations {
            if perturbation.id.is_empty() || perturbation.probability_per_mille > 1000 {
                return Err(SimError::Scenario(
                    "perturbation id must be non-empty and probability at most 1000".into(),
                ));
            }
            match perturbation.effect {
                PerturbationEffect::CloseDoor
                    if !self
                        .doors
                        .iter()
                        .any(|door| door.id == perturbation.target_id) =>
                {
                    return Err(SimError::Scenario(
                        "perturbation references an unknown door".into(),
                    ));
                }
                PerturbationEffect::CloseDoor => {}
                PerturbationEffect::ActivateHazard | PerturbationEffect::DeactivateHazard
                    if !self
                        .hazards
                        .iter()
                        .any(|hazard| hazard.id == perturbation.target_id) =>
                {
                    return Err(SimError::Scenario(
                        "perturbation references an unknown hazard".into(),
                    ));
                }
                PerturbationEffect::ActivateHazard | PerturbationEffect::DeactivateHazard => {}
                PerturbationEffect::MoveNpc
                    if self
                        .npc
                        .as_ref()
                        .is_none_or(|npc| npc.id != perturbation.target_id)
                        || perturbation
                            .destination
                            .is_none_or(|position| !self.in_bounds(position)) =>
                {
                    return Err(SimError::Scenario(
                        "NPC movement perturbation requires a known NPC and in-bounds destination"
                            .into(),
                    ));
                }
                PerturbationEffect::MoveNpc => {}
            }
        }
        let perturbation_ids = self
            .perturbations
            .iter()
            .map(|perturbation| perturbation.id.as_str())
            .collect::<BTreeSet<_>>();
        if perturbation_ids.len() != self.perturbations.len() {
            return Err(SimError::Scenario("perturbation ids must be unique".into()));
        }
        Ok(())
    }
    fn in_bounds(&self, p: Pos) -> bool {
        p.x >= 0 && p.x < self.width && p.y >= 0 && p.y < self.height
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub position: Pos,
    pub facing: Direction,
    pub health: i32,
    pub energy: i32,
    pub hydration: i32,
    pub inventory: Vec<String>,
    pub max_inventory: usize,
    pub max_inventory_weight: u32,
    #[serde(default)]
    pub status_effects: Vec<String>,
    pub alive: bool,
    pub escaped: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub event_id: Uuid,
    pub run_id: Uuid,
    pub step: u32,
    pub simulation_time: u32,
    pub timestamp: DateTime<Utc>,
    #[serde(rename = "type")]
    pub kind: String,
    pub message: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibleCell {
    pub relative_position: Pos,
    pub terrain: String,
    pub entities: Vec<VisibleEntity>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisibleEntity {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub state: Option<String>,
    pub name: Option<String>,
}
/// Body state available to a policy through an observation.
///
/// Deliberately excludes identity, absolute position, and terminal/evaluator
/// fields. Those belong to the authoritative world or researcher snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentObservation {
    pub facing: Direction,
    pub health: i32,
    pub energy: i32,
    pub hydration: i32,
    pub inventory: Vec<String>,
    pub max_inventory: usize,
    pub max_inventory_weight: u32,
    #[serde(default)]
    pub status_effects: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub protocol_version: u32,
    pub run_id: Uuid,
    pub step: u32,
    pub observation_mode: ObservationMode,
    pub agent: AgentObservation,
    pub goal: String,
    pub visible_cells: Vec<VisibleCell>,
    pub recent_events: Vec<String>,
    pub perception_note: Option<String>,
    pub allowed_action_types: Vec<String>,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Metrics {
    pub escaped: bool,
    pub alive: bool,
    pub steps_taken: u32,
    pub simulated_time: u32,
    pub final_health: i32,
    pub final_energy: i32,
    pub final_hydration: i32,
    pub invalid_actions: u32,
    pub repeated_invalid_actions: u32,
    pub unique_cells_visited: usize,
    pub useful_items_acquired: u32,
    pub carried_weight: u32,
    pub inventory_weight_capacity: u32,
    pub exploration_coverage: f64,
    pub action_diversity: usize,
    pub repeated_actions: u32,
    pub action_repetition_rate: f64,
    pub resource_efficiency: f64,
    pub hazard_damage_taken: i32,
    pub npc_interactions: u32,
    pub unnecessary_actions: u32,
    pub recovery_after_failure: bool,
    pub discovered_doors: usize,
    pub discovered_items: usize,
    pub discovered_hazards: usize,
    pub discovered_npcs: usize,
    pub first_discovery_steps: BTreeMap<String, u32>,
    pub milestones: BTreeMap<String, u32>,
    pub score_task_success: f64,
    pub score_health: f64,
    pub score_invalid_action_penalty: f64,
    pub score_step_penalty: f64,
    pub normalized_score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub observation: Observation,
    pub reward: i32,
    pub reward_breakdown: RewardBreakdown,
    pub done: bool,
    pub terminal_reason: Option<String>,
    pub events: Vec<Event>,
    pub step_number: u32,
    pub simulation_time: u32,
    pub metrics: Metrics,
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RewardBreakdown {
    pub baseline: i32,
    pub progress: i32,
    pub invalid_action_penalty: i32,
    pub hazard_penalty: i32,
    pub terminal: i32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldSnapshot {
    pub run_id: Uuid,
    pub step: u32,
    #[serde(default)]
    pub simulation_time: u32,
    #[serde(default)]
    pub goal: String,
    pub width: i32,
    pub height: i32,
    pub walls: Vec<Pos>,
    pub agent: Agent,
    pub items: Vec<Item>,
    pub doors: Vec<Door>,
    pub hazards: Vec<Hazard>,
    pub npc: Option<Npc>,
    pub containers: Vec<Container>,
    pub done: bool,
    pub terminal_reason: Option<String>,
    #[serde(default)]
    pub metrics: Metrics,
}

pub struct Environment {
    pub scenario: Scenario,
    pub run_id: Uuid,
    pub seed: u64,
    pub observation_mode: ObservationMode,
    pub agent: Agent,
    pub step: u32,
    pub simulation_time: u32,
    pub done: bool,
    pub terminal_reason: Option<String>,
    pub events: Vec<Event>,
    rng: StdRng,
    visited: BTreeSet<Pos>,
    invalid: u32,
    repeated_invalid: u32,
    last_invalid: Option<String>,
    useful_items: u32,
    rewarded_progress: BTreeSet<String>,
    hazard_damage: i32,
    npc_interactions: u32,
    discoveries: BTreeSet<String>,
    first_discovery_steps: BTreeMap<String, u32>,
    milestones: BTreeMap<String, u32>,
    action_types: BTreeSet<String>,
    last_action_type: Option<String>,
    repeated_actions: u32,
}
impl Environment {
    pub fn new(scenario: Scenario, seed: u64) -> Result<Self, SimError> {
        Self::new_with_observation_mode(scenario, seed, ObservationMode::Normal)
    }
    pub fn new_with_observation_mode(
        scenario: Scenario,
        seed: u64,
        observation_mode: ObservationMode,
    ) -> Result<Self, SimError> {
        scenario.validate()?;
        let spawn = scenario.spawn;
        let mut visited = BTreeSet::new();
        visited.insert(spawn);
        let mut environment = Self {
            scenario,
            run_id: Uuid::new_v4(),
            seed,
            observation_mode,
            agent: Agent {
                id: "agent_01".into(),
                name: "Explorer".into(),
                position: spawn,
                facing: Direction::East,
                health: 100,
                energy: 100,
                hydration: 100,
                inventory: vec![],
                max_inventory: 4,
                max_inventory_weight: 8,
                status_effects: vec![],
                alive: true,
                escaped: false,
            },
            step: 0,
            simulation_time: 0,
            done: false,
            terminal_reason: None,
            events: vec![],
            rng: StdRng::seed_from_u64(seed),
            visited,
            invalid: 0,
            repeated_invalid: 0,
            last_invalid: None,
            useful_items: 0,
            rewarded_progress: BTreeSet::new(),
            hazard_damage: 0,
            npc_interactions: 0,
            discoveries: BTreeSet::new(),
            first_discovery_steps: BTreeMap::new(),
            milestones: BTreeMap::new(),
            action_types: BTreeSet::new(),
            last_action_type: None,
            repeated_actions: 0,
        };
        environment.record_visible_discoveries(&mut vec![]);
        Ok(environment)
    }
    fn event(&mut self, kind: &str, message: impl Into<String>) -> Event {
        self.milestones.entry(kind.into()).or_insert(self.step);
        let e = Event {
            event_id: Uuid::new_v4(),
            run_id: self.run_id,
            step: self.step,
            simulation_time: self.simulation_time,
            timestamp: Utc::now(),
            kind: kind.into(),
            message: message.into(),
        };
        self.events.push(e.clone());
        e
    }
    fn at_door(&self, p: Pos) -> Option<&Door> {
        self.scenario.doors.iter().find(|x| x.position == p)
    }
    fn pickup_item(&self, p: Pos, id: Option<&str>) -> Option<&Item> {
        self.items_at(p)
            .find(|item| id.is_none_or(|id| item.id == id))
    }
    fn items_at(&self, p: Pos) -> impl Iterator<Item = &Item> {
        self.scenario.items.iter().filter(move |x| {
            x.position == p
                && !self.agent.inventory.contains(&x.id)
                && !self
                    .scenario
                    .containers
                    .iter()
                    .any(|c| !c.open && c.contents.contains(&x.id))
                && !self
                    .scenario
                    .npc
                    .as_ref()
                    .is_some_and(|npc| npc.inventory.contains(&x.id))
        })
    }
    fn adjacent(&self, p: Pos) -> bool {
        (self.agent.position.x - p.x).abs() + (self.agent.position.y - p.y).abs() == 1
    }
    fn carried_weight(&self) -> u32 {
        self.agent
            .inventory
            .iter()
            .filter_map(|item_id| self.scenario.items.iter().find(|item| &item.id == item_id))
            .map(|item| item.weight)
            .sum()
    }
    fn invalid(&mut self, key: String, msg: &str, out: &mut Vec<Event>) {
        self.invalid += 1;
        if self.last_invalid.as_ref() == Some(&key) {
            self.repeated_invalid += 1
        };
        self.last_invalid = Some(key);
        out.push(self.event("InvalidAction", msg));
    }
    fn record_visible_discoveries(&mut self, out: &mut Vec<Event>) {
        let entities = self
            .observe()
            .visible_cells
            .into_iter()
            .flat_map(|cell| cell.entities)
            .filter(|entity| matches!(entity.kind.as_str(), "door" | "item" | "hazard" | "npc"))
            .map(|entity| (entity.kind, entity.id))
            .collect::<Vec<_>>();
        for (kind, id) in entities {
            let discovery_key = format!("{kind}:{id}");
            if self.discoveries.insert(discovery_key.clone()) {
                self.first_discovery_steps.insert(discovery_key, self.step);
                out.push(self.event("EntityDiscovered", format!("Discovered {kind} {id}.")));
            }
        }
    }
    pub fn observe(&self) -> Observation {
        let flashlight_bonus = self
            .scenario
            .items
            .iter()
            .filter(|item| self.agent.inventory.contains(&item.id))
            .map(|item| item.vision_bonus)
            .max()
            .unwrap_or(0);
        let r = match self.observation_mode {
            ObservationMode::Minimal => 1,
            ObservationMode::Normal => self.scenario.vision_radius,
            ObservationMode::Rich => self.scenario.vision_radius + 1,
        } + flashlight_bonus;
        let mut cells = vec![];
        for y in self.agent.position.y - r..=self.agent.position.y + r {
            for x in self.agent.position.x - r..=self.agent.position.x + r {
                let p = Pos { x, y };
                if (x - self.agent.position.x).abs() + (y - self.agent.position.y).abs() > r
                    || !self.scenario.in_bounds(p)
                {
                    continue;
                };
                let mut entities = vec![];
                if let Some(d) = self.at_door(p) {
                    entities.push(VisibleEntity {
                        id: d.id.clone(),
                        kind: "door".into(),
                        state: Some(if d.open {
                            "open".into()
                        } else if d.locked {
                            "locked".into()
                        } else {
                            "closed".into()
                        }),
                        name: None,
                    })
                };
                for i in self.items_at(p) {
                    entities.push(VisibleEntity {
                        id: i.id.clone(),
                        kind: "item".into(),
                        state: None,
                        name: Some(i.name.clone()),
                    })
                }
                if let Some(c) = self.scenario.containers.iter().find(|c| c.position == p) {
                    entities.push(VisibleEntity {
                        id: c.id.clone(),
                        kind: "container".into(),
                        state: Some(if c.open {
                            "open".into()
                        } else {
                            "closed".into()
                        }),
                        name: None,
                    })
                };
                if let Some(h) = self
                    .scenario
                    .hazards
                    .iter()
                    .find(|h| h.position == p && h.active)
                {
                    entities.push(VisibleEntity {
                        id: h.id.clone(),
                        kind: "hazard".into(),
                        state: Some(if h.active {
                            "active".into()
                        } else {
                            "inactive".into()
                        }),
                        name: Some(h.kind.clone()),
                    })
                };
                if let Some(n) = &self.scenario.npc
                    && n.position == p
                {
                    entities.push(VisibleEntity {
                        id: n.id.clone(),
                        kind: "npc".into(),
                        // NPC trust is hidden internal state, not perception.
                        state: Some(n.disposition.clone()),
                        name: None,
                    })
                };
                let terrain = if self.scenario.walls.contains(&p) && self.at_door(p).is_none() {
                    "wall"
                } else {
                    "floor"
                };
                cells.push(VisibleCell {
                    relative_position: Pos {
                        x: x - self.agent.position.x,
                        y: y - self.agent.position.y,
                    },
                    terrain: terrain.into(),
                    entities,
                });
            }
        }
        Observation {
            protocol_version: PROTOCOL_VERSION,
            run_id: self.run_id,
            step: self.step,
            observation_mode: self.observation_mode,
            agent: AgentObservation {
                facing: self.agent.facing,
                health: self.agent.health,
                energy: self.agent.energy,
                hydration: self.agent.hydration,
                inventory: self.agent.inventory.clone(),
                max_inventory: self.agent.max_inventory,
                max_inventory_weight: self.agent.max_inventory_weight,
                status_effects: self.agent.status_effects.clone(),
            },
            goal: self.scenario.goal.clone(),
            visible_cells: cells,
            recent_events: self
                .events
                .iter()
                .rev()
                .take(5)
                .rev()
                .map(|e| e.message.clone())
                .collect(),
            perception_note: (self.observation_mode == ObservationMode::Rich)
                .then(|| "Extended local survey is enabled for this run.".into()),
            allowed_action_types: vec![
                "move", "inspect", "pickup", "drop", "use_item", "open", "close", "talk", "wait",
                "give", "rest",
            ]
            .into_iter()
            .map(String::from)
            .collect(),
        }
    }
    pub fn metrics(&self) -> Metrics {
        let escaped = self.terminal_reason.as_deref() == Some("escaped");
        let score_task_success = if escaped { 100.0 } else { 0.0 };
        let score_health = self.agent.health as f64 * 0.1;
        let score_invalid_action_penalty = self.invalid as f64 * 2.0;
        let score_step_penalty = self.step as f64 * 0.1;
        let score =
            score_task_success + score_health - score_invalid_action_penalty - score_step_penalty;
        Metrics {
            escaped,
            alive: self.agent.health > 0,
            steps_taken: self.step,
            simulated_time: self.simulation_time,
            final_health: self.agent.health,
            final_energy: self.agent.energy,
            final_hydration: self.agent.hydration,
            invalid_actions: self.invalid,
            repeated_invalid_actions: self.repeated_invalid,
            unique_cells_visited: self.visited.len(),
            useful_items_acquired: self.useful_items,
            carried_weight: self.carried_weight(),
            inventory_weight_capacity: self.agent.max_inventory_weight,
            exploration_coverage: (self.visited.len() as f64
                / ((self.scenario.width * self.scenario.height
                    - self
                        .scenario
                        .walls
                        .iter()
                        .filter(|wall| self.at_door(**wall).is_none())
                        .count() as i32)
                    .max(1) as f64))
                .clamp(0.0, 1.0),
            action_diversity: self.action_types.len(),
            repeated_actions: self.repeated_actions,
            action_repetition_rate: if self.step > 1 {
                self.repeated_actions as f64 / (self.step - 1) as f64
            } else {
                0.0
            },
            resource_efficiency: ((self.agent.health + self.agent.energy + self.agent.hydration)
                as f64
                / 300.0)
                .clamp(0.0, 1.0),
            hazard_damage_taken: self.hazard_damage,
            npc_interactions: self.npc_interactions,
            unnecessary_actions: self.step.saturating_sub(self.visited.len() as u32),
            recovery_after_failure: self.repeated_invalid == 0 && self.invalid > 0,
            discovered_doors: self
                .discoveries
                .iter()
                .filter(|discovery| discovery.starts_with("door:"))
                .count(),
            discovered_items: self
                .discoveries
                .iter()
                .filter(|discovery| discovery.starts_with("item:"))
                .count(),
            discovered_hazards: self
                .discoveries
                .iter()
                .filter(|discovery| discovery.starts_with("hazard:"))
                .count(),
            discovered_npcs: self
                .discoveries
                .iter()
                .filter(|discovery| discovery.starts_with("npc:"))
                .count(),
            first_discovery_steps: self.first_discovery_steps.clone(),
            milestones: self.milestones.clone(),
            score_task_success,
            score_health,
            score_invalid_action_penalty,
            score_step_penalty,
            normalized_score: score.clamp(0.0, 100.0),
        }
    }
    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot {
            run_id: self.run_id,
            step: self.step,
            simulation_time: self.simulation_time,
            goal: self.scenario.goal.clone(),
            width: self.scenario.width,
            height: self.scenario.height,
            walls: self.scenario.walls.clone(),
            agent: self.agent.clone(),
            items: self.scenario.items.clone(),
            doors: self.scenario.doors.clone(),
            hazards: self.scenario.hazards.clone(),
            npc: self.scenario.npc.clone(),
            containers: self.scenario.containers.clone(),
            done: self.done,
            terminal_reason: self.terminal_reason.clone(),
            metrics: self.metrics(),
        }
    }
    pub fn interrupt(&mut self, reason: &str, message: &str) -> Event {
        if self.done {
            return self.event("RunAlreadyTerminal", "Run is terminal.");
        }
        self.done = true;
        self.terminal_reason = Some(reason.into());
        self.event("RunInterrupted", message)
    }
    pub fn step(&mut self, action: Action) -> StepResult {
        let mut out = vec![];
        if self.done {
            out.push(self.event("RunAlreadyTerminal", "Run is terminal."));
            return StepResult {
                observation: self.observe(),
                reward: 0,
                reward_breakdown: RewardBreakdown::default(),
                done: true,
                terminal_reason: self.terminal_reason.clone(),
                events: out,
                step_number: self.step,
                simulation_time: self.simulation_time,
                metrics: self.metrics(),
            };
        }
        self.step += 1;
        let action_kind = match &action {
            Action::Move { .. } => "move",
            Action::Inspect { .. } => "inspect",
            Action::Pickup { .. } => "pickup",
            Action::Drop { .. } => "drop",
            Action::UseItem { .. } => "use_item",
            Action::Open { .. } => "open",
            Action::Close { .. } => "close",
            Action::Talk { .. } => "talk",
            Action::Give { .. } => "give",
            Action::Wait => "wait",
            Action::Rest => "rest",
        };
        if self.last_action_type.as_deref() == Some(action_kind) {
            self.repeated_actions += 1;
        }
        self.last_action_type = Some(action_kind.into());
        self.action_types.insert(action_kind.into());
        let invalid_before = self.invalid;
        let hazard_damage_before = self.hazard_damage;
        let action_energy_cost = match &action {
            Action::Move { .. } => 2,
            Action::Talk { .. } | Action::Give { .. } => 2,
            Action::Rest => 0,
            _ => 1,
        };
        let action_time_cost = match &action {
            Action::Talk { .. } | Action::Rest => 2,
            _ => 1,
        };
        self.simulation_time += action_time_cost;
        match action {
            Action::Move { direction } => {
                let target = self.agent.position.shifted(direction);
                let blocked = !self.scenario.in_bounds(target)
                    || self
                        .at_door(target)
                        .map(|door| !door.open)
                        .unwrap_or_else(|| self.scenario.walls.contains(&target));
                if blocked {
                    self.invalid(
                        format!("move:{:?}", direction),
                        "Movement blocked.",
                        &mut out,
                    )
                } else {
                    self.agent.position = target;
                    self.agent.facing = direction;
                    self.visited.insert(target);
                    out.push(self.event("AgentMoved", "Movement completed."));
                }
            }
            Action::Pickup { item_id } => {
                let item = self
                    .pickup_item(self.agent.position, item_id.as_deref())
                    .cloned();
                if let Some(item) = item {
                    if self.agent.inventory.len() >= self.agent.max_inventory
                        || self.carried_weight().saturating_add(item.weight)
                            > self.agent.max_inventory_weight
                    {
                        self.invalid("pickup:capacity".into(), "Inventory is full.", &mut out)
                    } else {
                        if self.rewarded_progress.insert(format!("item:{}", item.id)) {
                            self.useful_items += 1;
                            self.milestones
                                .insert(format!("item:{}", item.id), self.step);
                        }
                        self.agent.inventory.push(item.id);
                        out.push(self.event("ItemPickedUp", format!("Picked up {}.", item.name)));
                    }
                } else {
                    self.invalid("pickup:none".into(), "Nothing to pick up here.", &mut out)
                }
            }
            Action::UseItem { item_id } => {
                if let Some(i) = self
                    .scenario
                    .items
                    .iter()
                    .find(|i| i.id == item_id)
                    .cloned()
                    .filter(|_| self.agent.inventory.contains(&item_id))
                {
                    self.agent.health = (self.agent.health + i.health).min(100);
                    self.agent.energy = (self.agent.energy + i.energy).min(100);
                    self.agent.hydration = (self.agent.hydration + i.hydration).min(100);
                    self.agent
                        .status_effects
                        .retain(|effect| !i.cures_status_effects.contains(effect));
                    if i.consumable {
                        self.agent.inventory.retain(|x| x != &item_id);
                        self.scenario.items.retain(|x| x.id != item_id);
                        for container in &mut self.scenario.containers {
                            container.contents.retain(|id| id != &item_id);
                        }
                    };
                    out.push(self.event("ItemConsumed", format!("Used {}.", i.name)));
                } else {
                    self.invalid("use:unknown".into(), "That item is unavailable.", &mut out)
                }
            }
            Action::Drop { item_id } => {
                if let Some(idx) = self.agent.inventory.iter().position(|x| x == &item_id) {
                    self.agent.inventory.remove(idx);
                    if let Some(item) = self
                        .scenario
                        .items
                        .iter_mut()
                        .find(|item| item.id == item_id)
                    {
                        item.position = self.agent.position;
                        let item_name = item.name.clone();
                        out.push(self.event("ItemDropped", format!("Dropped {item_name}.")));
                    } else {
                        self.invalid(
                            "drop:unknown".into(),
                            "That item no longer exists.",
                            &mut out,
                        )
                    }
                } else {
                    self.invalid("drop:unknown".into(), "That item is not carried.", &mut out)
                }
            }
            Action::Open { target_id } => {
                let opened_container = if let Some(index) = self
                    .scenario
                    .containers
                    .iter()
                    .position(|c| c.id == target_id && c.position == self.agent.position)
                {
                    let key = self.scenario.containers[index].key_id.clone();
                    if self.scenario.containers[index].locked
                        && key
                            .as_ref()
                            .is_some_and(|required| !self.agent.inventory.contains(required))
                    {
                        self.invalid(
                            "open:container_locked".into(),
                            "The container is locked.",
                            &mut out,
                        );
                    } else {
                        self.scenario.containers[index].locked = false;
                        self.scenario.containers[index].open = true;
                        let milestone = format!("container:{}", self.scenario.containers[index].id);
                        if self.rewarded_progress.insert(milestone.clone()) {
                            self.milestones.insert(milestone, self.step);
                        }
                        out.push(self.event("ContainerOpened", "Container opened."));
                    }
                    true
                } else {
                    false
                };
                if !opened_container {
                    let can_open = self
                        .scenario
                        .doors
                        .iter()
                        .any(|door| door.id == target_id && self.adjacent(door.position));
                    if !can_open {
                        self.invalid("open:target".into(), "Door is not adjacent.", &mut out)
                    } else {
                        let idx = self
                            .scenario
                            .doors
                            .iter()
                            .position(|d| d.id == target_id)
                            .unwrap();
                        let key = self.scenario.doors[idx].key_id.clone();
                        if self.scenario.doors[idx].locked
                            && key
                                .as_ref()
                                .is_some_and(|k| !self.agent.inventory.contains(k))
                        {
                            self.invalid("open:locked".into(), "The door is locked.", &mut out)
                        } else {
                            self.scenario.doors[idx].locked = false;
                            self.scenario.doors[idx].open = true;
                            let milestone = format!("door:{}", self.scenario.doors[idx].id);
                            if self.rewarded_progress.insert(milestone.clone()) {
                                self.milestones.insert(milestone, self.step);
                            }
                            out.push(self.event("DoorOpened", "Door opened."));
                        }
                    }
                }
            }
            Action::Close { target_id } => {
                let agent_position = self.agent.position;
                if let Some(container) = self.scenario.containers.iter_mut().find(|container| {
                    container.id == target_id && container.position == agent_position
                }) {
                    container.open = false;
                    out.push(self.event("ContainerClosed", "Container closed."));
                } else if let Some(d) = self.scenario.doors.iter_mut().find(|d| {
                    d.id == target_id
                        && (agent_position.x - d.position.x).abs()
                            + (agent_position.y - d.position.y).abs()
                            == 1
                }) {
                    d.open = false;
                    out.push(self.event("DoorClosed", "Door closed."));
                } else {
                    self.invalid(
                        "close:target".into(),
                        "Door or container is unavailable.",
                        &mut out,
                    )
                }
            }
            Action::Talk { target_id, .. } => {
                if self.scenario.npc.as_ref().is_some_and(|n| {
                    n.id == target_id
                        && (n.position == self.agent.position || self.adjacent(n.position))
                }) {
                    let dialogue_index = self.npc_interactions as usize;
                    self.npc_interactions += 1;
                    let response = {
                        let npc = self.scenario.npc.as_mut().unwrap();
                        npc.trust = (npc.trust + 1).min(100);
                        npc.dialogue
                            .get(dialogue_index)
                            .cloned()
                            .unwrap_or_else(|| npc.hint.clone())
                    };
                    out.push(self.event("NpcSpoke", response));
                } else {
                    self.invalid("talk:target".into(), "Nobody is here.", &mut out)
                }
            }
            Action::Give { target_id, item_id } => {
                let npc_is_near = self.scenario.npc.as_ref().is_some_and(|npc| {
                    npc.id == target_id
                        && (npc.position == self.agent.position || self.adjacent(npc.position))
                });
                if !npc_is_near {
                    self.invalid(
                        "give:target".into(),
                        "Nobody is close enough to receive that.",
                        &mut out,
                    );
                } else if let Some(index) = self
                    .agent
                    .inventory
                    .iter()
                    .position(|item| item == &item_id)
                {
                    self.agent.inventory.remove(index);
                    let npc = self.scenario.npc.as_mut().unwrap();
                    npc.inventory.push(item_id.clone());
                    npc.trust = (npc.trust + 10).min(100);
                    out.push(self.event("ItemGiven", format!("Gave {item_id} to {target_id}.")));
                } else {
                    self.invalid("give:item".into(), "That item is not carried.", &mut out);
                }
            }
            Action::Inspect { target_id } => {
                let visible_target = target_id.as_ref().and_then(|target| {
                    self.observe()
                        .visible_cells
                        .into_iter()
                        .flat_map(|cell| cell.entities)
                        .find(|entity| &entity.id == target)
                });
                if target_id.is_none() {
                    let visible_count = self
                        .observe()
                        .visible_cells
                        .iter()
                        .map(|cell| cell.entities.len())
                        .sum::<usize>();
                    out.push(self.event(
                        "InspectionCompleted",
                        format!("Local inspection identifies {visible_count} visible entities."),
                    ));
                } else if let Some(entity) = visible_target {
                    let detail = entity
                        .state
                        .or(entity.name)
                        .unwrap_or_else(|| "no additional visible detail".into());
                    out.push(self.event(
                        "InspectionCompleted",
                        format!("{} is a visible {}: {detail}.", entity.id, entity.kind),
                    ));
                } else {
                    self.invalid(
                        "inspect:target".into(),
                        "That target is not currently visible.",
                        &mut out,
                    )
                }
            }
            Action::Rest => {
                if self
                    .scenario
                    .hazards
                    .iter()
                    .any(|hazard| hazard.active && hazard.position == self.agent.position)
                {
                    self.invalid("rest:hazard".into(), "Resting here is unsafe.", &mut out);
                } else {
                    self.agent.energy = (self.agent.energy + 15).min(100);
                    out.push(self.event("AgentRested", "You rest and recover some energy."));
                }
            }
            Action::Wait => out.push(self.event("ActionCompleted", "You wait briefly.")),
        }
        self.agent.energy = (self.agent.energy - action_energy_cost).max(0);
        self.agent.hydration = (self.agent.hydration - 1).max(0);
        if self.agent.hydration == 0 {
            self.agent.health = (self.agent.health - 2).max(0);
            self.agent.status_effects = vec!["dehydrated".into()];
            out.push(self.event("DehydrationDamage", "Severe dehydration causes damage."));
        }
        if self.rng.random_ratio(1, 8) {
            out.push(self.event(
                "AmbientSignal",
                "A distant ventilation system cycles through the room.",
            ));
        }
        let due_perturbations = self
            .scenario
            .perturbations
            .iter()
            .filter(|perturbation| perturbation.step == self.step)
            .cloned()
            .collect::<Vec<_>>();
        for perturbation in due_perturbations {
            if self.rng.random_range(0..1000) >= perturbation.probability_per_mille {
                continue;
            }
            match perturbation.effect {
                PerturbationEffect::CloseDoor => {
                    if let Some(door) = self
                        .scenario
                        .doors
                        .iter_mut()
                        .find(|door| door.id == perturbation.target_id)
                    {
                        door.open = false;
                        let door_id = door.id.clone();
                        out.push(self.event(
                            "PerturbationApplied",
                            format!("{door_id} closes unexpectedly."),
                        ));
                    }
                }
                PerturbationEffect::ActivateHazard | PerturbationEffect::DeactivateHazard => {
                    let active = matches!(perturbation.effect, PerturbationEffect::ActivateHazard);
                    if let Some(hazard) = self
                        .scenario
                        .hazards
                        .iter_mut()
                        .find(|hazard| hazard.id == perturbation.target_id)
                    {
                        hazard.active = active;
                        let hazard_id = hazard.id.clone();
                        out.push(self.event(
                            "PerturbationApplied",
                            format!(
                                "{hazard_id} is now {}.",
                                if active { "active" } else { "inactive" }
                            ),
                        ));
                    }
                }
                PerturbationEffect::MoveNpc => {
                    if let (Some(npc), Some(destination)) =
                        (self.scenario.npc.as_mut(), perturbation.destination)
                    {
                        npc.position = destination;
                        let npc_id = npc.id.clone();
                        out.push(self.event(
                            "NpcMoved",
                            format!("{npc_id} moves to another visible position."),
                        ));
                    }
                }
            }
        }
        if let Some(npc) = self.scenario.npc.as_mut()
            && !npc.patrol.is_empty()
        {
            let destination = npc.patrol[(self.step as usize - 1) % npc.patrol.len()];
            if destination != npc.position
                && destination.x >= 0
                && destination.x < self.scenario.width
                && destination.y >= 0
                && destination.y < self.scenario.height
                && !self.scenario.walls.contains(&destination)
            {
                npc.position = destination;
                let npc_id = npc.id.clone();
                out.push(self.event("NpcMoved", format!("{npc_id} follows a patrol route.")));
            }
        }
        if let Some(h) = self
            .scenario
            .hazards
            .iter()
            .find(|h| h.position == self.agent.position && h.active)
            .cloned()
        {
            self.agent.health = (self.agent.health - h.damage).max(0);
            self.agent.energy = (self.agent.energy - h.energy_drain).max(0);
            self.agent.hydration = (self.agent.hydration - h.hydration_drain).max(0);
            if let Some(status) = &h.status_effect
                && !self.agent.status_effects.contains(status)
            {
                self.agent.status_effects.push(status.clone());
            }
            self.hazard_damage += h.damage;
            out.push(self.event(
                "HazardTriggered",
                format!("The {} hazard affects the agent.", h.kind),
            ));
        }
        if self
            .scenario
            .doors
            .iter()
            .any(|door| door.is_exit && door.open && door.position == self.agent.position)
        {
            self.done = true;
            self.terminal_reason = Some("escaped".into());
            self.agent.escaped = true;
            out.push(self.event("GoalCompleted", "Escape achieved."));
        } else if self.agent.health <= 0 {
            self.done = true;
            self.terminal_reason = Some("dead".into());
            self.agent.alive = false;
            out.push(self.event("AgentDied", "Health reached zero."));
        } else if self.step >= self.scenario.max_steps {
            self.done = true;
            self.terminal_reason = Some("timeout".into());
            out.push(self.event("RunTimedOut", "Maximum steps reached."));
        } else if self
            .scenario
            .time_limit
            .is_some_and(|limit| self.simulation_time >= limit)
        {
            self.done = true;
            self.terminal_reason = Some("time_limit".into());
            out.push(self.event("RunTimedOut", "Simulation time limit reached."));
        }
        self.record_visible_discoveries(&mut out);
        let reward_breakdown = RewardBreakdown {
            baseline: -1,
            progress: self
                .rewarded_progress
                .iter()
                .filter(|key| self.milestones.get(*key) == Some(&self.step))
                .count() as i32
                * 5,
            invalid_action_penalty: -((self.invalid - invalid_before) as i32 * 2),
            hazard_penalty: -(self.hazard_damage - hazard_damage_before),
            terminal: if self.terminal_reason.as_deref() == Some("escaped") {
                100
            } else if self.done {
                -100
            } else {
                0
            },
        };
        let reward = reward_breakdown.baseline
            + reward_breakdown.progress
            + reward_breakdown.invalid_action_penalty
            + reward_breakdown.hazard_penalty
            + reward_breakdown.terminal;
        StepResult {
            observation: self.observe(),
            reward,
            reward_breakdown,
            done: self.done,
            terminal_reason: self.terminal_reason.clone(),
            events: out,
            step_number: self.step,
            simulation_time: self.simulation_time,
            metrics: self.metrics(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    fn s() -> Scenario {
        Scenario {
            id: "t".into(),
            name: "t".into(),
            version: 1,
            width: 5,
            height: 5,
            spawn: Pos { x: 1, y: 2 },
            max_steps: 20,
            time_limit: None,
            vision_radius: 1,
            goal: "go".into(),
            walls: vec![Pos { x: 0, y: 2 }, Pos { x: 4, y: 2 }],
            items: vec![Item {
                id: "k".into(),
                name: "key".into(),
                kind: "key".into(),
                position: Pos { x: 3, y: 2 },
                energy: 0,
                hydration: 0,
                health: 0,
                consumable: false,
                weight: 1,
                vision_bonus: 0,
                cures_status_effects: vec![],
            }],
            doors: vec![Door {
                id: "exit".into(),
                position: Pos { x: 4, y: 3 },
                locked: true,
                open: false,
                key_id: Some("k".into()),
                is_exit: true,
            }],
            hazards: vec![],
            npc: None,
            containers: vec![],
            perturbations: vec![],
            rooms: vec![Room {
                id: "test_room".into(),
                name: "Test Room".into(),
                min: Pos { x: 1, y: 1 },
                max: Pos { x: 3, y: 3 },
            }],
            routes: vec![Route {
                id: "test_route".into(),
                waypoints: vec![Pos { x: 1, y: 2 }, Pos { x: 2, y: 2 }],
            }],
        }
    }
    #[test]
    fn blocked_does_not_move() {
        let mut e = Environment::new(s(), 1).unwrap();
        let p = e.agent.position;
        e.step(Action::Move {
            direction: Direction::West,
        });
        assert_eq!(p, e.agent.position)
    }
    #[test]
    fn observation_hides_key() {
        let e = Environment::new(s(), 1).unwrap();
        assert!(
            !e.observe()
                .visible_cells
                .iter()
                .flat_map(|c| &c.entities)
                .any(|x| x.id == "k")
        )
    }
    #[test]
    fn observation_excludes_researcher_coordinates_and_npc_trust() {
        let mut scenario = s();
        scenario.npc = Some(Npc {
            id: "guide".into(),
            position: scenario.spawn,
            hint: "hello".into(),
            disposition: "friendly".into(),
            trust: 77,
            dialogue: vec![],
            patrol: vec![],
            inventory: vec![],
        });
        let env = Environment::new(scenario, 21).unwrap();
        let observation = serde_json::to_value(env.observe()).unwrap();
        assert!(observation["agent"].get("position").is_none());
        assert!(observation.get("current_room_id").is_none());
        assert!(!observation.to_string().contains("trust 77"));
        assert_eq!(env.snapshot().agent.position, env.agent.position);
    }
    #[test]
    fn discovery_metrics_only_record_entities_after_local_perception_reveals_them() {
        let mut env = Environment::new(s(), 1).unwrap();
        assert_eq!(env.metrics().discovered_items, 0);
        let first = env.step(Action::Move {
            direction: Direction::East,
        });
        assert_eq!(env.metrics().discovered_items, 1);
        assert_eq!(env.metrics().first_discovery_steps.get("item:k"), Some(&1));
        assert_eq!(env.metrics().milestones.get("EntityDiscovered"), Some(&1));
        assert!(
            first
                .events
                .iter()
                .any(|event| event.kind == "EntityDiscovered")
        );
        let second = env.step(Action::Wait);
        assert_eq!(env.metrics().discovered_items, 1);
        assert!(
            !second
                .events
                .iter()
                .any(|event| event.kind == "EntityDiscovered")
        );
    }
    #[test]
    fn hazard_can_end_episode_and_terminal_world_does_not_step() {
        let mut scenario = s();
        scenario.hazards.push(Hazard {
            id: "fire".into(),
            position: scenario.spawn,
            kind: "fire".into(),
            damage: 100,
            energy_drain: 0,
            hydration_drain: 0,
            status_effect: Some("burning".into()),
            active: true,
        });
        let mut env = Environment::new(scenario, 4).unwrap();
        let result = env.step(Action::Wait);
        assert!(result.done);
        assert_eq!(result.terminal_reason.as_deref(), Some("dead"));
        let terminal_step = env.step;
        let repeat = env.step(Action::Wait);
        assert_eq!(env.step, terminal_step);
        assert!(
            repeat
                .events
                .iter()
                .any(|event| event.kind == "RunAlreadyTerminal")
        );
    }
    #[test]
    fn resource_values_stay_in_bounds() {
        let mut env = Environment::new(s(), 5).unwrap();
        env.agent.energy = 1;
        env.agent.hydration = 1;
        for _ in 0..3 {
            env.step(Action::Move {
                direction: Direction::East,
            });
        }
        assert!((0..=100).contains(&env.agent.energy));
        assert!((0..=100).contains(&env.agent.hydration));
    }
    #[test]
    fn closed_container_hides_contents_until_opened() {
        let mut scenario = s();
        scenario.containers.push(Container {
            id: "locker".into(),
            position: Pos { x: 3, y: 2 },
            open: false,
            locked: false,
            key_id: None,
            contents: vec!["k".into()],
        });
        let mut env = Environment::new(scenario, 7).unwrap();
        env.step(Action::Move {
            direction: Direction::East,
        });
        env.step(Action::Move {
            direction: Direction::East,
        });
        assert!(
            !env.observe()
                .visible_cells
                .iter()
                .flat_map(|cell| &cell.entities)
                .any(|entity| entity.id == "k")
        );
        env.step(Action::Open {
            target_id: "locker".into(),
        });
        assert!(
            env.observe()
                .visible_cells
                .iter()
                .flat_map(|cell| &cell.entities)
                .any(|entity| entity.id == "k")
        );
        let close = env.step(Action::Close {
            target_id: "locker".into(),
        });
        assert!(
            close
                .events
                .iter()
                .any(|event| event.kind == "ContainerClosed")
        );
        assert!(
            !env.observe()
                .visible_cells
                .iter()
                .flat_map(|cell| &cell.entities)
                .any(|entity| entity.id == "k")
        );
    }
    #[test]
    fn locked_container_requires_its_own_key() {
        let mut scenario = s();
        scenario.containers.push(Container {
            id: "secure_locker".into(),
            position: scenario.spawn,
            open: false,
            locked: true,
            key_id: Some("k".into()),
            contents: vec![],
        });
        let mut env = Environment::new(scenario, 71).unwrap();
        let rejected = env.step(Action::Open {
            target_id: "secure_locker".into(),
        });
        assert!(
            rejected
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        env.agent.inventory.push("k".into());
        let opened = env.step(Action::Open {
            target_id: "secure_locker".into(),
        });
        assert!(
            opened
                .events
                .iter()
                .any(|event| event.kind == "ContainerOpened")
        );
        assert!(env.scenario.containers[0].open);
        assert!(!env.scenario.containers[0].locked);
    }
    #[test]
    fn all_colocated_visible_items_are_discovered_but_closed_contents_are_hidden() {
        let mut scenario = s();
        scenario.items[0].position = scenario.spawn;
        for id in ["second", "hidden"] {
            let mut item = scenario.items[0].clone();
            item.id = id.into();
            scenario.items.push(item);
        }
        scenario.containers.push(Container {
            id: "box".into(),
            position: scenario.spawn,
            open: false,
            locked: false,
            key_id: None,
            contents: vec!["hidden".into()],
        });
        let mut env = Environment::new(scenario, 42).unwrap();
        let visible_ids = |env: &Environment| {
            env.observe()
                .visible_cells
                .into_iter()
                .flat_map(|cell| cell.entities)
                .filter(|entity| entity.kind == "item")
                .map(|entity| entity.id)
                .collect::<Vec<_>>()
        };
        assert_eq!(visible_ids(&env), vec!["k", "second"]);
        assert_eq!(env.metrics().first_discovery_steps.get("item:k"), Some(&0));
        assert_eq!(
            env.metrics().first_discovery_steps.get("item:second"),
            Some(&0)
        );
        assert!(
            !env.metrics()
                .first_discovery_steps
                .contains_key("item:hidden")
        );
        env.step(Action::Open {
            target_id: "box".into(),
        });
        assert_eq!(visible_ids(&env), vec!["k", "second", "hidden"]);
        assert_eq!(
            env.metrics().first_discovery_steps.get("item:hidden"),
            Some(&1)
        );
    }

    #[test]
    fn repeated_pickups_and_door_opening_do_not_farm_progress() {
        let mut scenario = s();
        scenario.items[0].position = scenario.spawn;
        scenario.doors[0].position = Pos { x: 2, y: 2 };
        let mut env = Environment::new(scenario, 42).unwrap();
        assert_eq!(
            env.step(Action::Pickup { item_id: None })
                .reward_breakdown
                .progress,
            5
        );
        env.step(Action::Drop {
            item_id: "k".into(),
        });
        assert_eq!(
            env.step(Action::Pickup { item_id: None })
                .reward_breakdown
                .progress,
            0
        );
        assert_eq!(env.metrics().useful_items_acquired, 1);
        assert_eq!(env.metrics().milestones.get("item:k"), Some(&1));
        assert_eq!(
            env.step(Action::Open {
                target_id: "exit".into()
            })
            .reward_breakdown
            .progress,
            5
        );
        env.step(Action::Close {
            target_id: "exit".into(),
        });
        assert_eq!(
            env.step(Action::Open {
                target_id: "exit".into()
            })
            .reward_breakdown
            .progress,
            0
        );
        assert_eq!(env.metrics().milestones.get("door:exit"), Some(&4));
    }

    #[test]
    fn freestanding_closed_doors_block_and_unlocked_doors_need_no_key() {
        let mut scenario = s();
        scenario.doors[0].position = Pos { x: 2, y: 2 };
        scenario.doors[0].locked = false;
        scenario.doors[0].is_exit = false;
        let mut exit = scenario.doors[0].clone();
        exit.id = "other_exit".into();
        exit.position = Pos { x: 4, y: 3 };
        exit.is_exit = true;
        scenario.doors.push(exit);
        let mut env = Environment::new(scenario, 42).unwrap();
        let blocked = env.step(Action::Move {
            direction: Direction::East,
        });
        assert_eq!(env.agent.position, Pos { x: 1, y: 2 });
        assert!(
            blocked
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        env.step(Action::Open {
            target_id: "exit".into(),
        });
        assert!(env.scenario.doors[0].open);
        assert!(env.agent.inventory.is_empty());
        env.step(Action::Move {
            direction: Direction::East,
        });
        assert_eq!(env.agent.position, Pos { x: 2, y: 2 });
        assert!(!env.done);
    }

    #[test]
    fn explicit_pickup_selects_the_requested_colocated_item() {
        let mut scenario = s();
        scenario.items[0].position = scenario.spawn;
        let mut second = scenario.items[0].clone();
        second.id = "second".into();
        scenario.items.push(second);
        let mut env = Environment::new(scenario, 42).unwrap();
        env.step(Action::Pickup {
            item_id: Some("second".into()),
        });
        assert_eq!(env.agent.inventory, vec!["second"]);
        env.step(Action::Pickup { item_id: None });
        assert_eq!(env.agent.inventory, vec!["second", "k"]);
    }

    #[test]
    fn consumed_items_cannot_be_picked_up_or_used_again() {
        let mut scenario = s();
        let mut bandage = scenario.items[0].clone();
        bandage.id = "bandage".into();
        bandage.position = scenario.spawn;
        bandage.consumable = true;
        bandage.health = 10;
        scenario.items.push(bandage);
        let mut env = Environment::new(scenario, 42).unwrap();
        env.agent.health = 50;
        env.step(Action::Pickup {
            item_id: Some("bandage".into()),
        });
        env.step(Action::UseItem {
            item_id: "bandage".into(),
        });
        assert_eq!(env.agent.health, 60);
        assert!(!env.snapshot().items.iter().any(|item| item.id == "bandage"));
        let pickup = env.step(Action::Pickup {
            item_id: Some("bandage".into()),
        });
        assert!(
            pickup
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        let reuse = env.step(Action::UseItem {
            item_id: "bandage".into(),
        });
        assert!(
            reuse
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        assert_eq!(env.agent.health, 60);
    }

    #[test]
    fn deactivated_hazards_allow_rest() {
        let mut scenario = s();
        scenario.hazards.push(Hazard {
            id: "off".into(),
            position: scenario.spawn,
            kind: "fire".into(),
            damage: 10,
            energy_drain: 0,
            hydration_drain: 0,
            status_effect: None,
            active: false,
        });
        let mut env = Environment::new(scenario, 42).unwrap();
        env.agent.energy = 40;
        let result = env.step(Action::Rest);
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "AgentRested")
        );
        assert!(
            !result
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        assert_eq!(env.agent.energy, 55);
        assert_eq!(env.agent.health, 100);
    }

    #[test]
    fn weighted_inventory_flashlight_and_bandage_mechanics_work() {
        let mut scenario = s();
        scenario.items = vec![
            Item {
                id: "heavy".into(),
                name: "Heavy crate".into(),
                kind: "tool".into(),
                position: scenario.spawn,
                energy: 0,
                hydration: 0,
                health: 0,
                consumable: false,
                weight: 9,
                vision_bonus: 0,
                cures_status_effects: vec![],
            },
            Item {
                id: "flashlight".into(),
                name: "Flashlight".into(),
                kind: "flashlight".into(),
                position: Pos { x: 2, y: 2 },
                energy: 0,
                hydration: 0,
                health: 0,
                consumable: false,
                weight: 1,
                vision_bonus: 2,
                cures_status_effects: vec![],
            },
            Item {
                id: "bandage".into(),
                name: "Bandage".into(),
                kind: "bandage".into(),
                position: Pos { x: 2, y: 2 },
                energy: 0,
                hydration: 0,
                health: 20,
                consumable: true,
                weight: 1,
                vision_bonus: 0,
                cures_status_effects: vec!["burning".into()],
            },
        ];
        scenario.doors[0].key_id = None;
        let mut env = Environment::new(scenario, 72).unwrap();
        let rejected = env.step(Action::Pickup {
            item_id: Some("heavy".into()),
        });
        assert!(
            rejected
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );

        env.agent
            .inventory
            .extend(["flashlight".into(), "bandage".into()]);
        env.agent.health = 60;
        env.agent.status_effects.push("burning".into());
        let visible_with_light = env.observe().visible_cells.len();
        env.step(Action::UseItem {
            item_id: "bandage".into(),
        });
        assert_eq!(env.agent.health, 80);
        assert!(!env.agent.status_effects.contains(&"burning".into()));
        assert!(!env.agent.inventory.contains(&"bandage".into()));
        env.agent.inventory.retain(|item| item != "flashlight");
        assert!(visible_with_light > env.observe().visible_cells.len());
    }
    #[test]
    fn simulation_time_limit_and_reward_breakdown_are_explicit() {
        let mut scenario = s();
        scenario.time_limit = Some(2);
        let mut env = Environment::new(scenario, 73).unwrap();
        let result = env.step(Action::Rest);
        assert!(result.done);
        assert_eq!(result.terminal_reason.as_deref(), Some("time_limit"));
        assert_eq!(
            result.reward,
            result.reward_breakdown.baseline
                + result.reward_breakdown.progress
                + result.reward_breakdown.invalid_action_penalty
                + result.reward_breakdown.hazard_penalty
                + result.reward_breakdown.terminal
        );
        assert_eq!(result.reward_breakdown.terminal, -100);
        assert_eq!(result.metrics.action_diversity, 1);
        assert_eq!(result.metrics.repeated_actions, 0);
        assert_eq!(result.metrics.action_repetition_rate, 0.0);
        assert!((0.0..=1.0).contains(&result.metrics.exploration_coverage));
        assert!((0.0..=1.0).contains(&result.metrics.resource_efficiency));
    }
    #[test]
    fn repeated_action_metrics_capture_consecutive_policy_loops() {
        let mut env = Environment::new(s(), 11).unwrap();
        env.step(Action::Wait);
        let result = env.step(Action::Wait);
        assert_eq!(result.metrics.action_diversity, 1);
        assert_eq!(result.metrics.repeated_actions, 1);
        assert_eq!(result.metrics.action_repetition_rate, 1.0);
        assert_eq!(result.metrics.milestones.get("ActionCompleted"), Some(&1));
    }
    #[test]
    fn metrics_additions_remain_backward_compatible_with_saved_replays() {
        let metrics: Metrics = serde_json::from_str(r#"{"escaped":true,"steps_taken":4}"#).unwrap();
        assert!(metrics.escaped);
        assert_eq!(metrics.steps_taken, 4);
        assert!(metrics.first_discovery_steps.is_empty());
        assert_eq!(metrics.action_repetition_rate, 0.0);
    }
    #[test]
    fn typed_hazard_toggle_npc_dialogue_patrol_and_give_are_deterministic() {
        let mut scenario = s();
        scenario.npc = Some(Npc {
            id: "guide".into(),
            position: scenario.spawn,
            hint: "Fallback hint.".into(),
            disposition: "wary".into(),
            trust: 0,
            dialogue: vec!["First safe hint.".into(), "Second safe hint.".into()],
            patrol: vec![Pos { x: 2, y: 2 }],
            inventory: vec![],
        });
        scenario.hazards.push(Hazard {
            id: "gas".into(),
            position: scenario.spawn,
            kind: "toxic_gas".into(),
            damage: 3,
            energy_drain: 4,
            hydration_drain: 5,
            status_effect: Some("poisoned".into()),
            active: false,
        });
        scenario.perturbations.push(Perturbation {
            id: "gas_on".into(),
            step: 1,
            effect: PerturbationEffect::ActivateHazard,
            target_id: "gas".into(),
            destination: None,
            probability_per_mille: 1000,
        });
        scenario.perturbations.push(Perturbation {
            id: "gas_off".into(),
            step: 2,
            effect: PerturbationEffect::DeactivateHazard,
            target_id: "gas".into(),
            destination: None,
            probability_per_mille: 1000,
        });
        let mut env = Environment::new(scenario, 74).unwrap();
        env.agent.inventory.push("k".into());
        let first = env.step(Action::Give {
            target_id: "guide".into(),
            item_id: "k".into(),
        });
        assert!(first.events.iter().any(|event| event.kind == "ItemGiven"));
        assert!(
            first
                .events
                .iter()
                .any(|event| event.kind == "HazardTriggered")
        );
        assert!(first.events.iter().any(|event| event.kind == "NpcMoved"));
        assert!(env.agent.status_effects.contains(&"poisoned".into()));
        assert!(
            env.scenario
                .npc
                .as_ref()
                .unwrap()
                .inventory
                .contains(&"k".into())
        );
        assert_eq!(env.scenario.npc.as_ref().unwrap().trust, 10);

        let second = env.step(Action::Talk {
            target_id: "guide".into(),
            message: None,
        });
        assert!(
            second
                .events
                .iter()
                .any(|event| event.kind == "NpcSpoke" && event.message == "First safe hint.")
        );
        assert!(
            !second
                .events
                .iter()
                .any(|event| event.kind == "HazardTriggered")
        );
        let inspect = env.step(Action::Inspect {
            target_id: Some("guide".into()),
        });
        assert!(
            inspect
                .events
                .iter()
                .any(|event| event.kind == "InspectionCompleted"
                    && event.message.contains("wary")
                    && !event.message.contains("trust"))
        );
    }
    #[test]
    fn seeded_ambient_events_are_reproducible() {
        let mut first = Environment::new(s(), 99).unwrap();
        let mut second = Environment::new(s(), 99).unwrap();
        let first_events = (0..12)
            .flat_map(|_| first.step(Action::Wait).events)
            .map(|event| (event.kind, event.message))
            .collect::<Vec<_>>();
        let second_events = (0..12)
            .flat_map(|_| second.step(Action::Wait).events)
            .map(|event| (event.kind, event.message))
            .collect::<Vec<_>>();
        assert_eq!(first_events, second_events);
        assert!(first_events.iter().any(|(kind, _)| kind == "AmbientSignal"));
    }
    #[test]
    fn doors_can_open_from_any_adjacent_tile_but_not_remotely_close() {
        let mut env = Environment::new(s(), 8).unwrap();
        env.agent.position = Pos { x: 4, y: 2 };
        env.agent.inventory.push("k".into());
        env.step(Action::Open {
            target_id: "exit".into(),
        });
        assert!(env.scenario.doors[0].open);
        env.agent.position = Pos { x: 1, y: 1 };
        let result = env.step(Action::Close {
            target_id: "exit".into(),
        });
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        assert!(env.scenario.doors[0].open);
    }
    #[test]
    fn any_door_marked_as_an_exit_can_complete_an_alternate_route() {
        let mut scenario = s();
        scenario.doors[0].id = "emergency_hatch".into();
        scenario.doors[0].position = Pos { x: 2, y: 2 };
        scenario.doors[0].locked = false;
        scenario.doors[0].open = true;
        scenario.doors[0].key_id = None;
        let mut env = Environment::new(scenario, 5).unwrap();
        let result = env.step(Action::Move {
            direction: Direction::East,
        });
        assert!(result.done);
        assert_eq!(result.terminal_reason.as_deref(), Some("escaped"));
    }
    #[test]
    fn dropped_item_moves_to_agent_position_and_can_be_recovered() {
        let mut env = Environment::new(s(), 15).unwrap();
        env.step(Action::Move {
            direction: Direction::East,
        });
        env.step(Action::Move {
            direction: Direction::East,
        });
        env.step(Action::Pickup {
            item_id: Some("k".into()),
        });
        assert!(env.agent.inventory.contains(&"k".into()));
        env.step(Action::Move {
            direction: Direction::North,
        });
        let dropped_at = env.agent.position;
        env.step(Action::Drop {
            item_id: "k".into(),
        });
        assert!(!env.agent.inventory.contains(&"k".into()));
        assert_eq!(env.scenario.items[0].position, dropped_at);
        env.step(Action::Pickup {
            item_id: Some("k".into()),
        });
        assert!(env.agent.inventory.contains(&"k".into()));
    }
    #[test]
    fn rest_recovers_energy_but_is_rejected_on_a_hazard() {
        let mut env = Environment::new(s(), 16).unwrap();
        env.agent.energy = 40;
        env.step(Action::Rest);
        assert!(env.agent.energy > 40);
        env.scenario.hazards.push(Hazard {
            id: "unsafe".into(),
            position: env.agent.position,
            kind: "fire".into(),
            damage: 1,
            energy_drain: 0,
            hydration_drain: 0,
            status_effect: None,
            active: true,
        });
        let result = env.step(Action::Rest);
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
    }
    #[test]
    fn action_time_costs_are_reflected_in_metrics_and_step_results() {
        let mut env = Environment::new(s(), 20).unwrap();
        let wait = env.step(Action::Wait);
        assert_eq!(wait.step_number, 1);
        assert_eq!(wait.simulation_time, 1);
        let rest = env.step(Action::Rest);
        assert_eq!(rest.step_number, 2);
        assert_eq!(rest.simulation_time, 3);
        assert_eq!(rest.metrics.simulated_time, 3);
        assert_eq!(env.snapshot().simulation_time, 3);
    }
    #[test]
    fn inspect_never_confirms_a_hidden_target() {
        let mut env = Environment::new(s(), 17).unwrap();
        let result = env.step(Action::Inspect {
            target_id: Some("not_visible".into()),
        });
        assert!(
            result
                .events
                .iter()
                .any(|event| event.kind == "InvalidAction")
        );
        assert!(
            !result
                .observation
                .recent_events
                .iter()
                .any(|event| event.contains("not_visible"))
        );
    }
    #[test]
    fn observation_modes_control_only_local_perception_radius() {
        let mut scenario = s();
        scenario.vision_radius = 2;
        let minimal =
            Environment::new_with_observation_mode(scenario.clone(), 18, ObservationMode::Minimal)
                .unwrap();
        let normal =
            Environment::new_with_observation_mode(scenario.clone(), 18, ObservationMode::Normal)
                .unwrap();
        let rich =
            Environment::new_with_observation_mode(scenario, 18, ObservationMode::Rich).unwrap();
        assert!(minimal.observe().visible_cells.len() < normal.observe().visible_cells.len());
        assert!(normal.observe().visible_cells.len() < rich.observe().visible_cells.len());
        assert!(minimal.observe().perception_note.is_none());
        assert!(rich.observe().perception_note.is_some());
    }
    #[test]
    fn seeded_perturbations_change_world_state_reproducibly() {
        let mut scenario = s();
        scenario.perturbations.push(Perturbation {
            id: "close_exit".into(),
            step: 1,
            effect: PerturbationEffect::CloseDoor,
            target_id: "exit".into(),
            destination: None,
            probability_per_mille: 1000,
        });
        let mut first = Environment::new(scenario.clone(), 19).unwrap();
        let mut second = Environment::new(scenario, 19).unwrap();
        first.scenario.doors[0].open = true;
        second.scenario.doors[0].open = true;
        let first_events = first.step(Action::Wait).events;
        let second_events = second.step(Action::Wait).events;
        assert!(!first.scenario.doors[0].open);
        assert_eq!(
            first_events
                .iter()
                .map(|event| &event.kind)
                .collect::<Vec<_>>(),
            second_events
                .iter()
                .map(|event| &event.kind)
                .collect::<Vec<_>>()
        );
        assert!(
            first_events
                .iter()
                .any(|event| event.kind == "PerturbationApplied")
        );
    }
    #[test]
    fn configured_perturbations_can_vary_across_seeds() {
        let mut scenario = s();
        scenario.perturbations.push(Perturbation {
            id: "random_close".into(),
            step: 1,
            effect: PerturbationEffect::CloseDoor,
            target_id: "exit".into(),
            destination: None,
            probability_per_mille: 500,
        });
        let outcomes = (0..32)
            .map(|seed| {
                let mut environment = Environment::new(scenario.clone(), seed).unwrap();
                environment.scenario.doors[0].open = true;
                environment.step(Action::Wait);
                environment.scenario.doors[0].open
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(outcomes.len(), 2);
    }
    #[test]
    fn scenario_validation_rejects_invalid_references_and_duplicate_ids() {
        let mut invalid_key = s();
        invalid_key.doors[0].key_id = Some("missing".into());
        assert!(
            matches!(invalid_key.validate(), Err(SimError::Scenario(message)) if message.contains("unknown key"))
        );

        let mut duplicate = s();
        duplicate.items.push(Item {
            id: "k".into(),
            name: "duplicate".into(),
            kind: "tool".into(),
            position: Pos { x: 2, y: 2 },
            energy: 0,
            hydration: 0,
            health: 0,
            consumable: false,
            weight: 1,
            vision_bonus: 0,
            cures_status_effects: vec![],
        });
        assert!(
            matches!(duplicate.validate(), Err(SimError::Scenario(message)) if message.contains("unique"))
        );

        let mut blocked_spawn = s();
        blocked_spawn.walls.push(blocked_spawn.spawn);
        assert!(
            matches!(blocked_spawn.validate(), Err(SimError::Scenario(message)) if message.contains("spawn cannot"))
        );

        let mut duplicate_perturbation = s();
        duplicate_perturbation.perturbations = vec![
            Perturbation {
                id: "same".into(),
                step: 1,
                effect: PerturbationEffect::CloseDoor,
                target_id: "exit".into(),
                destination: None,
                probability_per_mille: 1,
            },
            Perturbation {
                id: "same".into(),
                step: 2,
                effect: PerturbationEffect::CloseDoor,
                target_id: "exit".into(),
                destination: None,
                probability_per_mille: 1,
            },
        ];
        assert!(
            matches!(duplicate_perturbation.validate(), Err(SimError::Scenario(message)) if message.contains("perturbation ids"))
        );

        let mut invalid_route = s();
        invalid_route.routes[0].waypoints = vec![Pos { x: 0, y: 2 }];
        assert!(
            matches!(invalid_route.validate(), Err(SimError::Scenario(message)) if message.contains("routes require"))
        );
    }
    proptest! {
        #[test]
        fn arbitrary_movement_keeps_resources_and_position_valid(directions in prop::collection::vec(0u8..4, 0..128)) {
            let mut env = Environment::new(s(), 123).unwrap();
            for index in directions {
                let direction = match index { 0 => Direction::North, 1 => Direction::South, 2 => Direction::East, _ => Direction::West };
                env.step(Action::Move { direction });
                prop_assert!((0..=100).contains(&env.agent.health));
                prop_assert!((0..=100).contains(&env.agent.energy));
                prop_assert!((0..=100).contains(&env.agent.hydration));
                prop_assert!(env.scenario.in_bounds(env.agent.position));
                if env.done { break; }
            }
        }

        #[test]
        fn terminal_environment_cannot_mutate_from_arbitrary_actions(actions in prop::collection::vec(0u8..3, 1..48)) {
            let mut scenario = s();
            scenario.hazards.push(Hazard { id: "fatal".into(), position: scenario.spawn, kind: "fire".into(), damage: 100, energy_drain: 0, hydration_drain: 0, status_effect: None, active: true });
            let mut env = Environment::new(scenario, 55).unwrap();
            env.step(Action::Wait);
            let terminal_step = env.step;
            let terminal_position = env.agent.position;
            for action in actions {
                match action { 0 => { env.step(Action::Wait); }, 1 => { env.step(Action::Move { direction: Direction::East }); }, _ => { env.step(Action::Rest); } }
                prop_assert_eq!(env.step, terminal_step);
                prop_assert_eq!(env.agent.position, terminal_position);
            }
        }
    }
}
