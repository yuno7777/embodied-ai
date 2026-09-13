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
pub struct WorldGeneratorConfig {
    pub generator_version: u32,
    pub min_width: i32,
    pub max_width: i32,
    pub min_height: i32,
    pub max_height: i32,
    pub max_attempts: u32,
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
        {
            return Err(SimError::Scenario("invalid world generator bounds".into()));
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
        // A divider creates two independently useful rooms joined by one deterministic corridor.
        // It keeps generation compact while preventing a generated world from degenerating into
        // an unrestricted open grid.
        let divider_x = width / 2;
        let key_in_right_room = rng.random_bool(0.5);
        let key_position = Pos {
            x: if key_in_right_room {
                rng.random_range(divider_x + 1..width - 1)
            } else {
                rng.random_range(1..divider_x)
            },
            y: rng.random_range(1..height - 1),
        };
        let hazard_position = loop {
            let position = Pos {
                x: rng.random_range(1..width - 1),
                y: rng.random_range(1..height - 1),
            };
            if position != key_position && position.y != middle_y && position.x != divider_x {
                break position;
            }
        };
        let supply_position = Pos {
            x: 1,
            y: (middle_y + 1).min(height - 2),
        };
        let spawn = Pos { x: 1, y: middle_y };
        let exit = Pos {
            x: width - 1,
            y: middle_y,
        };
        let mut walls = border_walls(width, height);
        walls.extend(
            (1..height - 1)
                .filter(|y| *y != middle_y)
                .map(|y| Pos { x: divider_x, y }),
        );
        walls.sort();
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
                    position: supply_position,
                    energy: 0,
                    hydration: 20,
                    health: 0,
                    consumable: true,
                    weight: 1,
                    vision_bonus: 0,
                    cures_status_effects: vec![],
                },
            ],
            doors: vec![
                Door {
                    id: "corridor_door".into(),
                    position: Pos {
                        x: divider_x,
                        y: middle_y,
                    },
                    locked: false,
                    open: true,
                    key_id: None,
                    is_exit: false,
                },
                Door {
                    id: "exit".into(),
                    position: exit,
                    locked: true,
                    open: false,
                    key_id: Some("exit_key".into()),
                    is_exit: true,
                },
            ],
            hazards: vec![Hazard {
                id: "hazard".into(),
                position: hazard_position,
                kind: "electrical".into(),
                damage: 5,
                energy_drain: 1,
                hydration_drain: 0,
                status_effect: Some("shocked".into()),
                active: true,
            }],
            npc: Some(Npc {
                id: "guide".into(),
                position: Pos {
                    x: width - 2,
                    y: (middle_y + 1).min(height - 2),
                },
                hint: "An exit key opens the marked exit.".into(),
                disposition: "neutral".into(),
                trust: 0,
                dialogue: vec!["Find the key, then open the exit.".into()],
                patrol: vec![],
                inventory: vec![],
            }),
            containers: vec![Container {
                id: "supply_case".into(),
                position: Pos {
                    x: 1,
                    y: (middle_y + 1).min(height - 2),
                },
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
            rooms: vec![
                Room {
                    id: "generated_left_room".into(),
                    name: "Generated Left Room".into(),
                    min: Pos { x: 1, y: 1 },
                    max: Pos {
                        x: divider_x - 1,
                        y: height - 2,
                    },
                },
                Room {
                    id: "generated_right_room".into(),
                    name: "Generated Right Room".into(),
                    min: Pos {
                        x: divider_x + 1,
                        y: 1,
                    },
                    max: Pos {
                        x: width - 2,
                        y: height - 2,
                    },
                },
            ],
            routes: vec![Route {
                id: "key_to_exit".into(),
                waypoints: vec![
                    spawn,
                    key_position,
                    Pos {
                        x: width - 2,
                        y: middle_y,
                    },
                    exit,
                ],
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
        let divider_x = scenario.width / 2;
        assert_eq!(scenario.rooms.len(), 2);
        assert!(
            scenario
                .walls
                .iter()
                .any(|wall| wall.x == divider_x && wall.y > 0 && wall.y < scenario.height - 1)
        );
        let corridor = scenario
            .doors
            .iter()
            .find(|door| door.id == "corridor_door")
            .unwrap();
        assert_eq!(corridor.position.x, divider_x);
        assert!(corridor.open && !corridor.locked);
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
