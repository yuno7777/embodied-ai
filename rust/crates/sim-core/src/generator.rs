//! Deterministic symbolic-world generation for Environment Domain 1.

use crate::{
    Container, Door, Hazard, Item, Npc, Perturbation, PerturbationEffect, Pos, Room, Route,
    Scenario, SimError,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, VecDeque};

pub const WORLD_GENERATOR_VERSION: u32 = 1;
pub const WORLD_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorldPartition {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldDistribution {
    pub train: std::ops::RangeInclusive<u64>,
    pub validation: std::ops::RangeInclusive<u64>,
    pub test: std::ops::RangeInclusive<u64>,
}

impl Default for WorldDistribution {
    fn default() -> Self {
        Self {
            train: 0..=7_999,
            validation: 8_000..=8_999,
            test: 9_000..=9_999,
        }
    }
}

impl WorldDistribution {
    pub fn validate(&self) -> Result<(), SimError> {
        let ranges = [&self.train, &self.validation, &self.test];
        if ranges.iter().any(|range| range.is_empty()) {
            return Err(SimError::Scenario("world partition cannot be empty".into()));
        }
        for (index, left) in ranges.iter().enumerate() {
            if ranges
                .iter()
                .skip(index + 1)
                .any(|right| left.start() <= right.end() && right.start() <= left.end())
            {
                return Err(SimError::Scenario(
                    "world partitions must not overlap".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn contains(&self, partition: WorldPartition, seed: u64) -> bool {
        match partition {
            WorldPartition::Train => self.train.contains(&seed),
            WorldPartition::Validation => self.validation.contains(&seed),
            WorldPartition::Test => self.test.contains(&seed),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct WorldGeneratorConfig {
    pub generator_version: u32,
    pub min_width: i32,
    pub max_width: i32,
    pub min_height: i32,
    pub max_height: i32,
    pub max_attempts: u32,
    /// Bounds on the compact corridor graph. Field defaults preserve manifests
    /// created before topology was independently configurable.
    #[serde(default = "default_min_rooms")]
    pub min_rooms: u8,
    #[serde(default = "default_max_rooms")]
    pub max_rooms: u8,
    #[serde(default = "default_hazard_kinds")]
    pub hazard_kinds: Vec<String>,
}

fn default_min_rooms() -> u8 {
    2
}

fn default_max_rooms() -> u8 {
    3
}

fn default_hazard_kinds() -> Vec<String> {
    vec!["electrical".into()]
}

impl Default for WorldGeneratorConfig {
    fn default() -> Self {
        Self {
            generator_version: WORLD_GENERATOR_VERSION,
            min_width: 11,
            max_width: 17,
            min_height: 7,
            max_height: 11,
            max_attempts: 16,
            min_rooms: default_min_rooms(),
            max_rooms: default_max_rooms(),
            hazard_kinds: default_hazard_kinds(),
        }
    }
}

impl WorldGeneratorConfig {
    pub fn validate(&self) -> Result<(), SimError> {
        if self.generator_version != WORLD_GENERATOR_VERSION {
            return Err(SimError::Scenario(
                "unsupported world generator version".into(),
            ));
        }
        if self.min_width < 5
            || self.min_height < 5
            || self.min_width > self.max_width
            || self.min_height > self.max_height
            || self.max_attempts == 0
            || self.min_rooms < 2
            || self.max_rooms > 3
            || self.min_rooms > self.max_rooms
            || self.hazard_kinds.is_empty()
        {
            return Err(SimError::Scenario("invalid world generator bounds".into()));
        }
        if self.max_rooms == 3 && self.min_width < 7 {
            return Err(SimError::Scenario(
                "three-room worlds require a minimum width of 7".into(),
            ));
        }
        if self
            .hazard_kinds
            .iter()
            .any(|kind| !matches!(kind.as_str(), "electrical" | "fire" | "toxic_gas"))
        {
            return Err(SimError::Scenario(
                "unsupported generated hazard kind".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorldValidation {
    pub geometry_valid: bool,
    pub spawn_valid: bool,
    pub required_key_reachable: bool,
    pub exit_reachable: bool,
    pub solvable: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldManifest {
    pub manifest_version: u32,
    pub generator_version: u32,
    pub seed: u64,
    pub generator_config: WorldGeneratorConfig,
    pub world_hash: String,
    pub dimensions: Pos,
    pub generation_attempt: u32,
    pub validation: WorldValidation,
    pub scenario: Scenario,
}

pub struct WorldGenerator {
    config: WorldGeneratorConfig,
}

impl WorldGenerator {
    pub fn new(config: WorldGeneratorConfig) -> Result<Self, SimError> {
        config.validate()?;
        Ok(Self { config })
    }

    pub fn generate(&self, seed: u64) -> Result<WorldManifest, SimError> {
        for attempt in 0..self.config.max_attempts {
            let scenario = self.candidate(seed, attempt);
            let validation = validate_generated_world(&scenario);
            if validation.solvable {
                let encoded = serde_json::to_vec(&scenario)
                    .map_err(|error| SimError::Scenario(error.to_string()))?;
                return Ok(WorldManifest {
                    manifest_version: WORLD_MANIFEST_VERSION,
                    generator_version: self.config.generator_version,
                    seed,
                    generator_config: self.config.clone(),
                    world_hash: stable_hash(&encoded),
                    dimensions: Pos {
                        x: scenario.width,
                        y: scenario.height,
                    },
                    generation_attempt: attempt,
                    validation,
                    scenario,
                });
            }
        }
        Err(SimError::Scenario(
            "generator exhausted deterministic attempts".into(),
        ))
    }

    fn candidate(&self, seed: u64, attempt: u32) -> Scenario {
        let derived_seed = seed ^ (u64::from(attempt).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let mut rng = StdRng::seed_from_u64(derived_seed);
        let width = rng.random_range(self.config.min_width..=self.config.max_width);
        let height = rng.random_range(self.config.min_height..=self.config.max_height);
        let middle_y = rng.random_range(1..height - 1);
        // Every divider has one corridor. Configuring the room-count range
        // makes topology a reproducible experimental variable rather than an
        // opaque source of random variation.
        let room_count = rng.random_range(self.config.min_rooms..=self.config.max_rooms);
        let divider_xs = match room_count {
            2 => vec![width / 2],
            3 => vec![width / 3, width * 2 / 3],
            _ => unreachable!("WorldGeneratorConfig validates supported room counts"),
        };
        let mut room_spans = vec![];
        let mut room_start = 1;
        for divider_x in &divider_xs {
            room_spans.push((room_start, *divider_x - 1));
            room_start = *divider_x + 1;
        }
        room_spans.push((room_start, width - 2));
        let spawn = random_position_in_span(&mut rng, room_spans[0], height);
        let key_room = rng.random_range(0..room_spans.len());
        let key_position = loop {
            let position = random_position_in_span(&mut rng, room_spans[key_room], height);
            if position != spawn {
                break position;
            }
        };
        let supply_position = loop {
            let position = random_position_in_span(&mut rng, room_spans[0], height);
            if position != spawn && position != key_position {
                break position;
            }
        };
        let hazard_position = loop {
            let position = Pos {
                x: rng.random_range(1..width - 1),
                y: rng.random_range(1..height - 1),
            };
            if position != spawn
                && position != key_position
                && position != supply_position
                && position.y != middle_y
                && !divider_xs.contains(&position.x)
            {
                break position;
            }
        };
        let water_position = loop {
            let room = rng.random_range(0..room_spans.len());
            let position = random_position_in_span(&mut rng, room_spans[room], height);
            if position != spawn
                && position != key_position
                && position != supply_position
                && position != hazard_position
            {
                break position;
            }
        };
        let npc_position = loop {
            let position = random_position_in_span(
                &mut rng,
                *room_spans.last().expect("at least one generated room"),
                height,
            );
            if position != spawn
                && position != key_position
                && position != supply_position
                && position != hazard_position
                && position != water_position
            {
                break position;
            }
        };
        let exit = Pos {
            x: width - 1,
            y: middle_y,
        };
        let selected_hazard =
            self.config.hazard_kinds[rng.random_range(0..self.config.hazard_kinds.len())].as_str();
        let (hazard_kind, damage, energy_drain, hydration_drain, status_effect) =
            match selected_hazard {
                "electrical" => ("electrical", 5, 1, 0, "shocked"),
                "fire" => ("fire", 6, 0, 1, "burning"),
                "toxic_gas" => ("toxic_gas", 3, 2, 2, "poisoned"),
                _ => unreachable!("WorldGeneratorConfig validates hazard kinds"),
            };
        let mut walls = border_walls(width, height);
        for divider_x in &divider_xs {
            walls.extend(
                (1..height - 1)
                    .filter(|y| *y != middle_y)
                    .map(|y| Pos { x: *divider_x, y }),
            );
        }
        walls.sort();
        let mut doors = divider_xs
            .iter()
            .enumerate()
            .map(|(index, divider_x)| Door {
                id: if index == 0 {
                    "corridor_door".into()
                } else {
                    format!("corridor_door_{}", index + 1)
                },
                position: Pos {
                    x: *divider_x,
                    y: middle_y,
                },
                locked: false,
                open: true,
                key_id: None,
                is_exit: false,
            })
            .collect::<Vec<_>>();
        doors.push(Door {
            id: "exit".into(),
            position: exit,
            locked: true,
            open: false,
            key_id: Some("exit_key".into()),
            is_exit: true,
        });
        let room_labels = ["Left", "Middle", "Right"];
        let rooms = room_spans
            .iter()
            .enumerate()
            .map(|(index, (min_x, max_x))| Room {
                id: format!("generated_{}_room", room_labels[index].to_lowercase()),
                name: format!("Generated {} Room", room_labels[index]),
                min: Pos { x: *min_x, y: 1 },
                max: Pos {
                    x: *max_x,
                    y: height - 2,
                },
            })
            .collect::<Vec<_>>();
        let mut key_to_exit = vec![spawn, key_position];
        key_to_exit.extend(divider_xs.iter().map(|divider_x| Pos {
            x: *divider_x,
            y: middle_y,
        }));
        key_to_exit.push(Pos {
            x: width - 2,
            y: middle_y,
        });
        key_to_exit.push(exit);
        Scenario {
            id: format!("generated_grid_v1_{seed}_{attempt}"),
            name: format!("Generated Grid {seed}"),
            version: 1,
            width,
            height,
            spawn,
            max_steps: (width * height * 2) as u32,
            time_limit: Some((width * height * 3) as u32),
            vision_radius: 2,
            goal: "Find the exit key and leave through the marked exit.".into(),
            walls,
            items: vec![
                Item {
                    id: "exit_key".into(),
                    name: "Exit key".into(),
                    kind: "key".into(),
                    position: key_position,
                    energy: 0,
                    hydration: 0,
                    health: 0,
                    consumable: false,
                    weight: 1,
                    vision_bonus: 0,
                    cures_status_effects: vec![],
                },
                Item {
                    id: "water".into(),
                    name: "Water flask".into(),
                    kind: "water".into(),
                    position: water_position,
                    energy: 0,
                    hydration: 20,
                    health: 0,
                    consumable: true,
                    weight: 1,
                    vision_bonus: 0,
                    cures_status_effects: vec![],
                },
            ],
            doors,
            hazards: vec![Hazard {
                id: "hazard".into(),
                position: hazard_position,
                kind: hazard_kind.into(),
                damage,
                energy_drain,
                hydration_drain,
                status_effect: Some(status_effect.into()),
                active: true,
            }],
            npc: Some(Npc {
                id: "guide".into(),
                position: npc_position,
                hint: "An exit key opens the marked exit.".into(),
                disposition: "neutral".into(),
                trust: 0,
                dialogue: vec!["Find the key, then open the exit.".into()],
                patrol: vec![],
                inventory: vec![],
            }),
            containers: vec![Container {
                id: "supply_case".into(),
                position: supply_position,
                open: true,
                locked: false,
                key_id: None,
                contents: vec![],
            }],
            perturbations: vec![Perturbation {
                id: "hazard_cooldown".into(),
                step: rng.random_range(4..=12),
                effect: PerturbationEffect::DeactivateHazard,
                target_id: "hazard".into(),
                destination: None,
                probability_per_mille: 1000,
            }],
            rooms,
            routes: vec![Route {
                id: "key_to_exit".into(),
                waypoints: key_to_exit,
            }],
        }
    }
}

fn border_walls(width: i32, height: i32) -> Vec<Pos> {
    let mut walls = BTreeSet::new();
    for x in 0..width {
        walls.insert(Pos { x, y: 0 });
        walls.insert(Pos { x, y: height - 1 });
    }
    for y in 0..height {
        walls.insert(Pos { x: 0, y });
        walls.insert(Pos { x: width - 1, y });
    }
    walls.into_iter().collect()
}

fn random_position_in_span(rng: &mut StdRng, span: (i32, i32), height: i32) -> Pos {
    Pos {
        x: rng.random_range(span.0..=span.1),
        y: rng.random_range(1..height - 1),
    }
}

pub fn validate_generated_world(scenario: &Scenario) -> WorldValidation {
    let geometry_valid = scenario.validate().is_ok();
    let spawn_valid = scenario.spawn.x > 0
        && scenario.spawn.x < scenario.width - 1
        && scenario.spawn.y > 0
        && scenario.spawn.y < scenario.height - 1;
    let exit = scenario.doors.iter().find(|door| door.is_exit);
    let required_key_reachable = exit
        .and_then(|door| door.key_id.as_deref())
        .and_then(|key_id| scenario.items.iter().find(|item| item.id == key_id))
        .is_some_and(|key| reachable(scenario, scenario.spawn, key.position, false));
    let exit_reachable = exit.is_some_and(|door| {
        adjacent_positions(door.position)
            .into_iter()
            .any(|position| reachable(scenario, scenario.spawn, position, false))
    });
    WorldValidation {
        geometry_valid,
        spawn_valid,
        required_key_reachable,
        exit_reachable,
        solvable: geometry_valid && spawn_valid && required_key_reachable && exit_reachable,
    }
}

fn reachable(scenario: &Scenario, start: Pos, target: Pos, allow_doors: bool) -> bool {
    let mut queue = VecDeque::from([start]);
    let mut seen = BTreeSet::from([start]);
    while let Some(position) = queue.pop_front() {
        if position == target {
            return true;
        }
        for next in adjacent_positions(position) {
            let in_bounds =
                next.x >= 0 && next.x < scenario.width && next.y >= 0 && next.y < scenario.height;
            let wall = scenario.walls.contains(&next);
            let door = scenario.doors.iter().any(|door| door.position == next);
            if in_bounds && (!wall || (allow_doors && door)) && seen.insert(next) {
                queue.push_back(next);
            }
        }
    }
    false
}

fn adjacent_positions(position: Pos) -> [Pos; 4] {
    [
        Pos {
            x: position.x,
            y: position.y - 1,
        },
        Pos {
            x: position.x,
            y: position.y + 1,
        },
        Pos {
            x: position.x - 1,
            y: position.y,
        },
        Pos {
            x: position.x + 1,
            y: position.y,
        },
    ]
}

fn stable_hash(bytes: &[u8]) -> String {
    let mut value: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("fnv1a64:{value:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_and_config_produce_the_same_manifest() {
        let generator = WorldGenerator::new(WorldGeneratorConfig::default()).unwrap();
        let left = generator.generate(42).unwrap();
        let right = generator.generate(42).unwrap();
        assert_eq!(left.world_hash, right.world_hash);
        assert_eq!(left.generation_attempt, right.generation_attempt);
        assert_eq!(
            serde_json::to_value(left.scenario).unwrap(),
            serde_json::to_value(right.scenario).unwrap()
        );
    }

    #[test]
    fn different_seeds_vary_generated_worlds_and_remain_solvable() {
        let generator = WorldGenerator::new(WorldGeneratorConfig::default()).unwrap();
        let worlds = (0..8)
            .map(|seed| generator.generate(seed).unwrap())
            .collect::<Vec<_>>();
        assert!(worlds.iter().all(|world| world.validation.solvable));
        assert!(
            worlds
                .windows(2)
                .any(|pair| pair[0].world_hash != pair[1].world_hash)
        );
    }

    #[test]
    fn generated_worlds_have_separated_rooms_and_a_traversable_corridor() {
        let world = WorldGenerator::new(WorldGeneratorConfig::default())
            .unwrap()
            .generate(42)
            .unwrap();
        let scenario = &world.scenario;
        assert!(matches!(scenario.rooms.len(), 2 | 3));
        let corridors = scenario
            .doors
            .iter()
            .filter(|door| !door.is_exit)
            .collect::<Vec<_>>();
        assert_eq!(corridors.len(), scenario.rooms.len() - 1);
        for corridor in corridors {
            assert!(corridor.open && !corridor.locked);
            assert!(scenario.walls.iter().any(|wall| {
                wall.x == corridor.position.x && wall.y > 0 && wall.y < scenario.height - 1
            }));
        }
        assert!(reachable(
            scenario,
            scenario.spawn,
            scenario.items[0].position,
            false
        ));
        assert!(matches!(
            scenario.perturbations.as_slice(),
            [Perturbation { id, target_id, effect: PerturbationEffect::DeactivateHazard, probability_per_mille: 1000, .. }]
                if id == "hazard_cooldown" && target_id == "hazard"
        ));
    }

    #[test]
    fn generated_worlds_vary_their_room_graph_topology() {
        let generator = WorldGenerator::new(WorldGeneratorConfig::default()).unwrap();
        let room_counts = (0..64)
            .map(|seed| generator.generate(seed).unwrap().scenario.rooms.len())
            .collect::<BTreeSet<_>>();
        assert_eq!(room_counts, BTreeSet::from([2, 3]));
    }

    #[test]
    fn generated_worlds_vary_entity_placements_without_colliding_critical_entities() {
        let generator = WorldGenerator::new(WorldGeneratorConfig::default()).unwrap();
        let worlds = (0..32)
            .map(|seed| generator.generate(seed).unwrap().scenario)
            .collect::<Vec<_>>();
        let spawns = worlds
            .iter()
            .map(|world| world.spawn)
            .collect::<BTreeSet<_>>();
        let supply_positions = worlds
            .iter()
            .map(|world| world.containers[0].position)
            .collect::<BTreeSet<_>>();
        let npc_positions = worlds
            .iter()
            .map(|world| world.npc.as_ref().unwrap().position)
            .collect::<BTreeSet<_>>();
        assert!(spawns.len() > 1);
        assert!(supply_positions.len() > 1);
        assert!(npc_positions.len() > 1);
        for world in worlds {
            let positions = [
                world.spawn,
                world.items[0].position,
                world.items[1].position,
                world.hazards[0].position,
                world.containers[0].position,
                world.npc.as_ref().unwrap().position,
            ];
            assert_eq!(
                positions.len(),
                positions.into_iter().collect::<BTreeSet<_>>().len()
            );
        }
    }

    #[test]
    fn configured_room_count_selects_a_reproducible_topology_family() {
        for room_count in [2, 3] {
            let generator = WorldGenerator::new(WorldGeneratorConfig {
                min_rooms: room_count,
                max_rooms: room_count,
                ..WorldGeneratorConfig::default()
            })
            .unwrap();
            for seed in 0..16 {
                let world = generator.generate(seed).unwrap();
                assert_eq!(world.scenario.rooms.len(), usize::from(room_count));
                assert_eq!(world.generator_config.min_rooms, room_count);
                assert_eq!(world.generator_config.max_rooms, room_count);
            }
        }
    }

    #[test]
    fn configured_hazard_family_is_deterministic_and_manifested() {
        let config = WorldGeneratorConfig {
            hazard_kinds: vec!["fire".into()],
            ..WorldGeneratorConfig::default()
        };
        let world = WorldGenerator::new(config.clone())
            .unwrap()
            .generate(7)
            .unwrap();
        assert_eq!(world.generator_config.hazard_kinds, vec!["fire"]);
        assert_eq!(world.scenario.hazards[0].kind, "fire");
        assert_eq!(
            world.scenario.hazards[0].status_effect.as_deref(),
            Some("burning")
        );
        assert!(
            WorldGenerator::new(WorldGeneratorConfig {
                hazard_kinds: vec!["lava".into()],
                ..config
            })
            .is_err()
        );
    }

    #[test]
    fn generator_rejects_unsupported_or_inverted_room_count_ranges() {
        for (min_rooms, max_rooms) in [(1, 2), (2, 4), (3, 2)] {
            assert!(
                WorldGenerator::new(WorldGeneratorConfig {
                    min_rooms,
                    max_rooms,
                    ..WorldGeneratorConfig::default()
                })
                .is_err()
            );
        }
    }

    #[test]
    fn generator_config_rejects_unknown_serialized_fields() {
        let config = serde_json::json!({
            "generator_version": 1,
            "min_width": 9,
            "max_width": 11,
            "min_height": 7,
            "max_height": 9,
            "max_attempts": 16,
            "min_rooms": 2,
            "max_rooms": 3,
            "hazard_kinds": ["electrical"],
            "min_wdith": 9
        });
        assert!(serde_json::from_value::<WorldGeneratorConfig>(config).is_err());
    }

    #[test]
    fn generator_rejects_a_three_room_config_that_cannot_make_nonempty_rooms() {
        assert!(
            WorldGenerator::new(WorldGeneratorConfig {
                min_width: 6,
                max_width: 8,
                min_rooms: 3,
                max_rooms: 3,
                ..WorldGeneratorConfig::default()
            })
            .is_err()
        );
        assert!(
            WorldGenerator::new(WorldGeneratorConfig {
                min_width: 7,
                max_width: 7,
                min_rooms: 3,
                max_rooms: 3,
                ..WorldGeneratorConfig::default()
            })
            .unwrap()
            .generate(1)
            .is_ok()
        );
    }

    #[test]
    fn default_world_partitions_are_disjoint_and_hold_out_test_seeds() {
        let distribution = WorldDistribution::default();
        distribution.validate().unwrap();
        assert!(distribution.contains(WorldPartition::Train, 42));
        assert!(distribution.contains(WorldPartition::Validation, 8_000));
        assert!(distribution.contains(WorldPartition::Test, 9_000));
        assert!(!distribution.contains(WorldPartition::Train, 9_000));
        assert!(!distribution.contains(WorldPartition::Validation, 9_000));
    }
}
