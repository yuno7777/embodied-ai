use axum::{
    Json, Router,
    extract::{
        DefaultBodyLimit, Path, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{Method, StatusCode, header::CONTENT_TYPE},
    routing::{get, post},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sim_core::{
    Action, Environment, Event, Observation, ObservationMode, RewardConfig, Scenario, StepResult,
    WorldSnapshot,
    generator::{
        WorldDistribution, WorldGenerator, WorldGeneratorConfig, WorldManifest, WorldPartition,
    },
};
use std::{
    collections::HashMap,
    path::{Path as FsPath, PathBuf},
    sync::Arc,
    time::Instant,
};
use tokio::sync::{Mutex, broadcast};
use tower_http::{cors::CorsLayer, services::ServeDir};
use uuid::Uuid;

struct StrictAction(Action);
impl<'de> serde::Deserialize<'de> for StrictAction {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        let object = value
            .as_object()
            .ok_or_else(|| serde::de::Error::custom("action must be a JSON object"))?;
        let action_type = object
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| serde::de::Error::custom("action type is required"))?;
        let allowed: &[&str] = match action_type {
            "move" => &["type", "direction"],
            "inspect" => &["type", "target_id"],
            "pickup" | "drop" | "use_item" => &["type", "item_id"],
            "open" | "close" => &["type", "target_id"],
            "talk" => &["type", "target_id", "message"],
            "give" => &["type", "target_id", "item_id"],
            "wait" | "rest" => &["type"],
            _ => return Err(serde::de::Error::custom("unknown action type")),
        };
        if let Some(field) = object
            .keys()
            .find(|field| !allowed.contains(&field.as_str()))
        {
            return Err(serde::de::Error::custom(format!(
                "unknown action field: {field}"
            )));
        }
        for field in ["target_id", "item_id"] {
            if object
                .get(field)
                .and_then(serde_json::Value::as_str)
                .is_some_and(|value| value.chars().count() > 128)
            {
                return Err(serde::de::Error::custom(format!(
                    "{field} must be at most 128 characters"
                )));
            }
        }
        if object
            .get("message")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|value| value.chars().count() > 500)
        {
            return Err(serde::de::Error::custom(
                "message must be at most 500 characters",
            ));
        }
        serde_json::from_value(value)
            .map(Self)
            .map_err(serde::de::Error::custom)
    }
}

struct RunRecord {
    env: Environment,
    /// The exact generated input world, when this run did not come from the catalog.
    /// Keeping it alongside live state makes a replay self-contained.
    world_manifest: Option<WorldManifest>,
    paused: bool,
    history: Vec<WorldSnapshot>,
    observations: Vec<Observation>,
    decisions: Vec<DecisionRecord>,
    actions: Vec<Action>,
    control: ControlStats,
    controller: Controller,
    control_phase: ControlPhase,
    control_since: Instant,
}
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Controller {
    #[default]
    Provider,
    Manual,
}
#[derive(Clone, Copy)]
enum ControlPhase {
    Provider,
    Manual,
    Paused,
    Finished,
}
impl From<Controller> for ControlPhase {
    fn from(controller: Controller) -> Self {
        match controller {
            Controller::Provider => Self::Provider,
            Controller::Manual => Self::Manual,
        }
    }
}
impl RunRecord {
    fn control_snapshot_at(&self, now: Instant) -> ControlStats {
        let mut control = self.control.clone();
        let elapsed = now
            .saturating_duration_since(self.control_since)
            .as_millis() as u64;
        let counter = match self.control_phase {
            ControlPhase::Provider => &mut control.provider_control_ms,
            ControlPhase::Manual => &mut control.manual_control_ms,
            ControlPhase::Paused => &mut control.paused_ms,
            ControlPhase::Finished => return control,
        };
        *counter = counter.saturating_add(elapsed);
        control
    }
    fn control_snapshot(&self) -> ControlStats {
        self.control_snapshot_at(Instant::now())
    }
    fn transition_control_at(&mut self, phase: ControlPhase, now: Instant) {
        self.control = self.control_snapshot_at(now);
        self.control_phase = phase;
        self.control_since = now;
    }
    fn transition_control(&mut self, phase: ControlPhase) {
        self.transition_control_at(phase, Instant::now());
    }
}
#[derive(Serialize)]
struct RunStatus {
    paused: bool,
    done: bool,
    terminal_reason: Option<String>,
    step: u32,
}
#[derive(Clone)]
struct AppState {
    runs: Arc<Mutex<HashMap<Uuid, RunRecord>>>,
    updates: broadcast::Sender<(Uuid, String)>,
}
impl Default for AppState {
    fn default() -> Self {
        let (updates, _) = broadcast::channel(128);
        Self {
            runs: Arc::default(),
            updates,
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CreateRun {
    scenario_id: Option<String>,
    generated_world: Option<GenerateWorld>,
    seed: Option<u64>,
    max_steps: Option<u32>,
    observation_mode: Option<ObservationMode>,
    reward_config: Option<RewardConfig>,
    #[serde(default)]
    controller: Controller,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateWorld {
    seed: u64,
    #[serde(default)]
    config: WorldGeneratorConfig,
    /// Optional assertion that this seed belongs to the named built-in split.
    /// Custom experiment splits remain manifest-defined rather than coerced
    /// into the built-in distribution.
    partition: Option<WorldPartition>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StopRun {
    reason: StopReason,
}
#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum StopReason {
    ClientTimeout,
    TokenBudgetExhausted,
}
#[derive(Clone, Serialize, Deserialize)]
struct TokenUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cached_tokens: Option<u64>,
    total_tokens: Option<u64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PlannerMetadata {
    name: String,
    expanded_nodes: Option<u32>,
    planning_time_ms: Option<f64>,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct AgentMetadata {
    confidence: Option<f64>,
    value_estimate: Option<f64>,
    policy_entropy: Option<f64>,
    planner: Option<PlannerMetadata>,
}
#[derive(Clone, Serialize, Deserialize)]
struct DecisionRecord {
    step: u32,
    action: Action,
    decision_summary: String,
    provider: String,
    model: Option<String>,
    latency_ms: Option<u64>,
    #[serde(default)]
    token_usage: Option<TokenUsage>,
    #[serde(default)]
    agent_metadata: Option<AgentMetadata>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DecisionInput {
    action: StrictAction,
    decision_summary: String,
    provider: String,
    model: Option<String>,
    latency_ms: Option<u64>,
    token_usage: Option<TokenUsage>,
    agent_metadata: Option<AgentMetadata>,
}
#[derive(Clone, Default, Serialize, Deserialize)]
#[serde(default)]
struct ControlStats {
    paused_ms: u64,
    provider_control_ms: u64,
    manual_control_ms: u64,
    provider_steps: u32,
    manual_steps: u32,
    simulation_latency_us: Vec<u64>,
}
#[derive(serde::Serialize, serde::Deserialize)]
struct Replay {
    #[serde(default)]
    seed: u64,
    #[serde(default)]
    scenario_id: String,
    #[serde(default)]
    scenario_version: u32,
    #[serde(default)]
    max_steps: u32,
    #[serde(default)]
    observation_mode: ObservationMode,
    #[serde(default)]
    engine_version: String,
    /// Present for procedural episodes so they can be replayed after catalog changes.
    #[serde(default)]
    world_manifest: Option<WorldManifest>,
    #[serde(default)]
    reward_config: RewardConfig,
    snapshot: WorldSnapshot,
    events: Vec<Event>,
    #[serde(default)]
    timeline: Vec<WorldSnapshot>,
    #[serde(default)]
    observations: Vec<Observation>,
    #[serde(default)]
    decisions: Vec<DecisionRecord>,
    #[serde(default)]
    actions: Vec<Action>,
    #[serde(default)]
    control: ControlStats,
}

struct PersistenceInput<'a> {
    env: &'a Environment,
    world_manifest: Option<&'a WorldManifest>,
    timeline: &'a [WorldSnapshot],
    observations: &'a [Observation],
    decisions: &'a [DecisionRecord],
    actions: &'a [Action],
    control: ControlStats,
}

#[derive(Serialize)]
struct BenchmarkSummary {
    total_runs: usize,
    completed_runs: usize,
    escaped_runs: usize,
    success_rate: f64,
    mean_steps: f64,
    mean_simulated_time: f64,
    mean_normalized_score: f64,
    mean_final_health: f64,
    mean_invalid_actions: f64,
    mean_hazard_damage: f64,
    total_provider_calls: usize,
    mean_provider_latency_ms: Option<f64>,
    total_input_tokens: Option<u64>,
    total_output_tokens: Option<u64>,
}

fn replay_for(
    env: &Environment,
    world_manifest: Option<&WorldManifest>,
    timeline: &[WorldSnapshot],
    observations: &[Observation],
    decisions: &[DecisionRecord],
    actions: &[Action],
    control: ControlStats,
) -> Replay {
    Replay {
        seed: env.seed,
        scenario_id: env.scenario.id.clone(),
        scenario_version: env.scenario.version,
        max_steps: env.scenario.max_steps,
        observation_mode: env.observation_mode,
        engine_version: "rust-v1".into(),
        world_manifest: world_manifest.cloned(),
        reward_config: env.reward_config.clone(),
        snapshot: env.snapshot(),
        events: env.events.clone(),
        timeline: timeline.to_vec(),
        observations: observations.to_vec(),
        decisions: decisions.to_vec(),
        actions: actions.to_vec(),
        control,
    }
}

fn bundled_scenario() -> Scenario {
    serde_json::from_str(include_str!(
        "../../../../scenarios/survival_room/scenario.rust.json"
    ))
    .expect("bundled scenario is valid")
}
fn scenario_directory() -> PathBuf {
    std::env::var_os("SIM_SCENARIO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("../scenarios"))
}
fn scenario_catalog_from(root: &FsPath) -> Result<Vec<Scenario>, String> {
    let entries = std::fs::read_dir(root).map_err(|error| error.to_string())?;
    let mut scenarios = entries
        .flatten()
        .filter_map(|entry| {
            entry
                .file_type()
                .ok()
                .filter(|kind| kind.is_dir())
                .map(|_| entry.path())
        })
        .filter_map(|directory| std::fs::read(directory.join("scenario.rust.json")).ok())
        .map(|raw| serde_json::from_slice::<Scenario>(&raw).map_err(|error| error.to_string()))
        .collect::<Result<Vec<_>, _>>()?;
    for configured in &scenarios {
        configured.validate().map_err(|error| error.to_string())?;
    }
    scenarios.sort_by(|left, right| left.id.cmp(&right.id));
    if scenarios.is_empty() {
        return Err("scenario catalog is empty".into());
    }
    Ok(scenarios)
}
fn scenario_catalog() -> Vec<Scenario> {
    scenario_catalog_from(&scenario_directory()).unwrap_or_else(|_| vec![bundled_scenario()])
}
fn scenario() -> Scenario {
    scenario_catalog()
        .into_iter()
        .find(|configured| configured.id == "survival_room")
        .unwrap_or_else(bundled_scenario)
}
fn persist(
    env: &Environment,
    world_manifest: Option<&WorldManifest>,
    timeline: &[WorldSnapshot],
    observations: &[Observation],
    decisions: &[DecisionRecord],
    actions: &[Action],
    control: ControlStats,
) -> Result<(), String> {
    let directory = std::path::Path::new("../data/runs");
    persist_to(
        directory,
        PersistenceInput {
            env,
            world_manifest,
            timeline,
            observations,
            decisions,
            actions,
            control,
        },
    )
}
fn persist_to(directory: &FsPath, input: PersistenceInput<'_>) -> Result<(), String> {
    std::fs::create_dir_all(directory)
        .map_err(|error| format!("cannot create replay directory: {error}"))?;
    let records = input
        .env
        .events
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| format!("cannot serialize event log: {error}"))?
        .join("\n");
    std::fs::write(
        directory.join(format!("{}.jsonl", input.env.run_id)),
        format!("{records}\n"),
    )
    .map_err(|error| format!("cannot write event log: {error}"))?;
    let replay = replay_for(
        input.env,
        input.world_manifest,
        input.timeline,
        input.observations,
        input.decisions,
        input.actions,
        input.control,
    );
    let encoded = serde_json::to_vec_pretty(&replay)
        .map_err(|error| format!("cannot serialize replay: {error}"))?;
    std::fs::write(
        directory.join(format!("{}.replay.json", input.env.run_id)),
        encoded,
    )
    .map_err(|error| format!("cannot write replay: {error}"))?;
    Ok(())
}
fn persistence_error(error: String) -> (StatusCode, String) {
    tracing::error!(%error, "replay persistence failed");
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        "replay persistence failed".into(),
    )
}
fn saved_replay(id: Uuid) -> Option<Replay> {
    let path = std::path::Path::new("../data/runs").join(format!("{id}.replay.json"));
    std::fs::read(path)
        .ok()
        .and_then(|raw| serde_json::from_slice(&raw).ok())
}
fn saved_replays() -> Vec<Replay> {
    let Ok(entries) = std::fs::read_dir("../data/runs") else {
        return vec![];
    };
    entries
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .ends_with(".replay.json")
        })
        .filter_map(|entry| std::fs::read(entry.path()).ok())
        .filter_map(|raw| serde_json::from_slice(&raw).ok())
        .collect()
}
async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({"status":"ok","protocol_version":1,"engine":"rust"}))
}
async fn scenarios() -> Json<Vec<Scenario>> {
    Json(scenario_catalog())
}
async fn scenario_by_id(Path(id): Path<String>) -> Result<Json<Scenario>, StatusCode> {
    scenario_catalog()
        .into_iter()
        .find(|configured| configured.id == id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}
async fn generate_world(
    Json(request): Json<GenerateWorld>,
) -> Result<Json<WorldManifest>, (StatusCode, String)> {
    validate_requested_partition(&request).map_err(|message| (StatusCode::BAD_REQUEST, message))?;
    let generator = WorldGenerator::new(request.config)
        .map_err(|error| (StatusCode::BAD_REQUEST, error.to_string()))?;
    generator
        .generate(request.seed)
        .map(Json)
        .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))
}
fn validate_requested_partition(request: &GenerateWorld) -> Result<(), String> {
    let Some(partition) = request.partition else {
        return Ok(());
    };
    if WorldDistribution::default().contains(partition, request.seed) {
        Ok(())
    } else {
        Err(format!(
            "seed {} is not in the requested built-in world partition",
            request.seed
        ))
    }
}
async fn world_partition(Path(seed): Path<u64>) -> Result<Json<WorldPartition>, StatusCode> {
    let distribution = WorldDistribution::default();
    [
        WorldPartition::Train,
        WorldPartition::Validation,
        WorldPartition::Test,
    ]
    .into_iter()
    .find(|partition| distribution.contains(*partition, seed))
    .map(Json)
    .ok_or(StatusCode::NOT_FOUND)
}
async fn create(
    State(state): State<AppState>,
    body: Option<Json<CreateRun>>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let request = body.map(|body| body.0).unwrap_or(CreateRun {
        scenario_id: None,
        generated_world: None,
        seed: None,
        max_steps: None,
        observation_mode: None,
        reward_config: None,
        controller: Controller::Provider,
    });
    if request.scenario_id.is_some() && request.generated_world.is_some() {
        return Err((
            StatusCode::BAD_REQUEST,
            "choose either scenario_id or generated_world".into(),
        ));
    }
    let generated_manifest = request
        .generated_world
        .as_ref()
        .map(|generated| {
            validate_requested_partition(generated)
                .map_err(|message| (StatusCode::BAD_REQUEST, message))?;
            WorldGenerator::new(generated.config.clone())
                .and_then(|generator| generator.generate(generated.seed))
                .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))
        })
        .transpose()?;
    let mut configured_scenario = match generated_manifest.as_ref() {
        Some(manifest) => manifest.scenario.clone(),
        None => match request.scenario_id.as_deref() {
            Some(id) => scenario_catalog()
                .into_iter()
                .find(|configured| configured.id == id)
                .ok_or_else(|| (StatusCode::BAD_REQUEST, format!("unknown scenario: {id}")))?,
            None => scenario(),
        },
    };
    if let Some(max_steps) = request.max_steps {
        configured_scenario.max_steps = max_steps;
    }
    let reward_config = request.reward_config.unwrap_or_default();
    let env = Environment::new_with_reward_config(
        configured_scenario,
        request.seed.unwrap_or(42),
        request.observation_mode.unwrap_or_default(),
        reward_config.clone(),
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let id = env.run_id;
    let observation = env.observe();
    let snapshot = env.snapshot();
    state.runs.lock().await.insert(
        id,
        RunRecord {
            env,
            world_manifest: generated_manifest.clone(),
            paused: false,
            history: vec![snapshot.clone()],
            observations: vec![observation.clone()],
            decisions: vec![],
            actions: vec![],
            control: ControlStats::default(),
            controller: request.controller,
            control_phase: request.controller.into(),
            control_since: Instant::now(),
        },
    );
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"run_created","snapshot":snapshot.clone()}).to_string(),
    ));
    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::json!({"run_id":id,"observation":observation,"snapshot":snapshot,"world_manifest":generated_manifest,"reward_config":reward_config}),
        ),
    ))
}
async fn runs(State(state): State<AppState>) -> Json<Vec<WorldSnapshot>> {
    Json(
        state
            .runs
            .lock()
            .await
            .values()
            .map(|r| r.env.snapshot())
            .collect(),
    )
}
async fn snapshot(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshot>, StatusCode> {
    state
        .runs
        .lock()
        .await
        .get(&id)
        .map(|r| Json(r.env.snapshot()))
        .ok_or(StatusCode::NOT_FOUND)
}
async fn run_status(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<RunStatus>, StatusCode> {
    state
        .runs
        .lock()
        .await
        .get(&id)
        .map(|run| {
            Json(RunStatus {
                paused: run.paused,
                done: run.env.done,
                terminal_reason: run.env.terminal_reason.clone(),
                step: run.env.step,
            })
        })
        .ok_or(StatusCode::NOT_FOUND)
}
async fn observation(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<Observation>, StatusCode> {
    state
        .runs
        .lock()
        .await
        .get(&id)
        .map(|run| Json(run.env.observe()))
        .ok_or(StatusCode::NOT_FOUND)
}
async fn events(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<Vec<Event>>, StatusCode> {
    state
        .runs
        .lock()
        .await
        .get(&id)
        .map(|r| Json(r.env.events.clone()))
        .ok_or(StatusCode::NOT_FOUND)
}
async fn replay(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<Replay>, StatusCode> {
    let current = state.runs.lock().await.get(&id).map(|r| Replay {
        seed: r.env.seed,
        scenario_id: r.env.scenario.id.clone(),
        scenario_version: r.env.scenario.version,
        max_steps: r.env.scenario.max_steps,
        observation_mode: r.env.observation_mode,
        engine_version: "rust-v1".into(),
        world_manifest: r.world_manifest.clone(),
        reward_config: r.env.reward_config.clone(),
        snapshot: r.env.snapshot(),
        events: r.env.events.clone(),
        timeline: r.history.clone(),
        observations: r.observations.clone(),
        decisions: r.decisions.clone(),
        actions: r.actions.clone(),
        control: r.control_snapshot(),
    });
    current
        .or_else(|| saved_replay(id))
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}
async fn restore_replay(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<(StatusCode, Json<serde_json::Value>), (StatusCode, String)> {
    let saved = saved_replay(id).ok_or((StatusCode::NOT_FOUND, "replay not found".into()))?;
    let generated_manifest = saved.world_manifest.clone();
    let mut configured = match generated_manifest.as_ref() {
        Some(manifest) => {
            manifest
                .scenario
                .validate()
                .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))?;
            if saved.scenario_id != manifest.scenario.id
                || saved.scenario_version != manifest.scenario.version
            {
                return Err((
                    StatusCode::CONFLICT,
                    "replay metadata does not match its generated world manifest".into(),
                ));
            }
            manifest.scenario.clone()
        }
        None => {
            let configured = scenario();
            if saved.scenario_id != configured.id || saved.scenario_version != configured.version {
                return Err((
                    StatusCode::CONFLICT,
                    "replay scenario version does not match the loaded scenario".into(),
                ));
            }
            configured
        }
    };
    if saved.snapshot.done {
        return Err((
            StatusCode::CONFLICT,
            "terminal replays cannot be resumed".into(),
        ));
    }
    if saved.actions.len() != saved.snapshot.step as usize {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "replay lacks a complete structured action stream".into(),
        ));
    }
    if saved.max_steps > 0 {
        configured.max_steps = saved.max_steps;
    }
    let mut env = Environment::new_with_reward_config(
        configured,
        saved.seed,
        saved.observation_mode,
        saved.reward_config,
    )
    .map_err(|error| (StatusCode::UNPROCESSABLE_ENTITY, error.to_string()))?;
    let mut history = vec![env.snapshot()];
    let mut observations = vec![env.observe()];
    for action in &saved.actions {
        let result = env.step(action.clone());
        history.push(env.snapshot());
        observations.push(result.observation);
    }
    let mut rebuilt = serde_json::to_value(env.snapshot()).unwrap_or_default();
    let mut expected = serde_json::to_value(&saved.snapshot).unwrap_or_default();
    rebuilt.as_object_mut().map(|value| value.remove("run_id"));
    expected.as_object_mut().map(|value| value.remove("run_id"));
    fn normalize_floats(value: &mut serde_json::Value) {
        match value {
            serde_json::Value::Number(number) if !number.is_i64() && !number.is_u64() => {
                if let Some(float) = number.as_f64() {
                    *value = serde_json::json!((float * 1_000_000_000.0).round() / 1_000_000_000.0);
                }
            }
            serde_json::Value::Array(values) => values.iter_mut().for_each(normalize_floats),
            serde_json::Value::Object(values) => values.values_mut().for_each(normalize_floats),
            _ => {}
        }
    }
    normalize_floats(&mut rebuilt);
    normalize_floats(&mut expected);
    if rebuilt != expected {
        return Err((
            StatusCode::CONFLICT,
            "replay action stream does not reconstruct its final snapshot".into(),
        ));
    }
    let new_id = env.run_id;
    let snapshot = env.snapshot();
    let observation = env.observe();
    let record = RunRecord {
        env,
        world_manifest: generated_manifest,
        paused: false,
        history,
        observations,
        decisions: saved.decisions,
        actions: saved.actions,
        control: saved.control,
        controller: Controller::Provider,
        control_phase: ControlPhase::Provider,
        control_since: Instant::now(),
    };
    state.runs.lock().await.insert(new_id, record);
    Ok((
        StatusCode::CREATED,
        Json(
            serde_json::json!({"run_id":new_id,"restored_from":id,"snapshot":snapshot,"observation":observation}),
        ),
    ))
}
async fn archived_replays() -> Json<Vec<Replay>> {
    Json(saved_replays())
}
async fn benchmarks(State(state): State<AppState>) -> Json<BenchmarkSummary> {
    let mut records = state
        .runs
        .lock()
        .await
        .values()
        .map(|run| {
            let snapshot = run.env.snapshot();
            (snapshot.run_id, (snapshot, run.decisions.clone()))
        })
        .collect::<HashMap<Uuid, (WorldSnapshot, Vec<DecisionRecord>)>>();
    for replay in saved_replays() {
        records
            .entry(replay.snapshot.run_id)
            .or_insert((replay.snapshot, replay.decisions));
    }
    let total_runs = records.len();
    let completed = records
        .into_values()
        .filter(|(snapshot, _)| snapshot.done)
        .collect::<Vec<_>>();
    let count = completed.len() as f64;
    let average = |value: fn(&WorldSnapshot) -> f64| {
        if completed.is_empty() {
            0.0
        } else {
            completed
                .iter()
                .map(|(snapshot, _)| value(snapshot))
                .sum::<f64>()
                / count
        }
    };
    let escaped_runs = completed
        .iter()
        .filter(|(snapshot, _)| snapshot.terminal_reason.as_deref() == Some("escaped"))
        .count();
    let decisions = completed
        .iter()
        .flat_map(|(_, decisions)| decisions)
        .collect::<Vec<_>>();
    let latencies = decisions
        .iter()
        .filter_map(|decision| decision.latency_ms)
        .collect::<Vec<_>>();
    let token_total = |extract: fn(&TokenUsage) -> Option<u64>| {
        let values = decisions
            .iter()
            .filter_map(|decision| decision.token_usage.as_ref().and_then(extract))
            .collect::<Vec<_>>();
        (!values.is_empty()).then(|| values.into_iter().sum())
    };
    Json(BenchmarkSummary {
        total_runs,
        completed_runs: completed.len(),
        escaped_runs,
        success_rate: if completed.is_empty() {
            0.0
        } else {
            escaped_runs as f64 / count
        },
        mean_steps: average(|snapshot| snapshot.step as f64),
        mean_simulated_time: average(|snapshot| snapshot.metrics.simulated_time as f64),
        mean_normalized_score: average(|snapshot| snapshot.metrics.normalized_score),
        mean_final_health: average(|snapshot| snapshot.metrics.final_health as f64),
        mean_invalid_actions: average(|snapshot| snapshot.metrics.invalid_actions as f64),
        mean_hazard_damage: average(|snapshot| snapshot.metrics.hazard_damage_taken as f64),
        total_provider_calls: decisions.len(),
        mean_provider_latency_ms: (!latencies.is_empty()).then(|| {
            latencies.iter().map(|latency| *latency as f64).sum::<f64>() / latencies.len() as f64
        }),
        total_input_tokens: token_total(|usage| usage.input_tokens),
        total_output_tokens: token_total(|usage| usage.output_tokens),
    })
}
async fn pause(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if run.env.done {
        return Err(StatusCode::CONFLICT);
    }
    if !run.paused {
        run.transition_control(ControlPhase::Paused);
    }
    run.paused = true;
    let _ = state
        .updates
        .send((id, serde_json::json!({"type":"run_paused"}).to_string()));
    Ok(Json(serde_json::json!({"paused":true})))
}
async fn resume(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if run.env.done {
        return Err(StatusCode::CONFLICT);
    }
    if run.paused {
        run.transition_control(run.controller.into());
    }
    run.paused = false;
    let _ = state
        .updates
        .send((id, serde_json::json!({"type":"run_resumed"}).to_string()));
    Ok(Json(serde_json::json!({"paused":false})))
}
async fn abort(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshot>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if run.env.done {
        return Ok(Json(run.env.snapshot()));
    }
    run.transition_control(ControlPhase::Finished);
    run.env
        .interrupt("aborted", "Run was aborted by the researcher.");
    let snapshot = run.env.snapshot();
    run.history.push(snapshot.clone());
    run.observations.push(run.env.observe());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(|error| persistence_error(error).0)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"run_completed","snapshot":snapshot.clone()}).to_string(),
    ));
    Ok(Json(snapshot))
}
async fn provider_error(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
) -> Result<Json<WorldSnapshot>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if run.env.done {
        return Ok(Json(run.env.snapshot()));
    }
    run.transition_control(ControlPhase::Finished);
    run.env.interrupt(
        "provider_error",
        "The provider became unavailable; partial run state was preserved.",
    );
    let snapshot = run.env.snapshot();
    run.history.push(snapshot.clone());
    run.observations.push(run.env.observe());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(|error| persistence_error(error).0)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"run_completed","snapshot":snapshot.clone()}).to_string(),
    ));
    Ok(Json(snapshot))
}
async fn stop_run(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(input): Json<StopRun>,
) -> Result<Json<WorldSnapshot>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    let (reason, message) = match input.reason {
        StopReason::ClientTimeout => (
            "client_timeout",
            "The orchestration wall-clock time budget was exhausted.",
        ),
        StopReason::TokenBudgetExhausted => (
            "token_budget_exhausted",
            "The provider token budget was exhausted.",
        ),
    };
    if run.env.done {
        return Ok(Json(run.env.snapshot()));
    }
    run.transition_control(ControlPhase::Finished);
    run.env.interrupt(reason, message);
    let snapshot = run.env.snapshot();
    run.history.push(snapshot.clone());
    run.observations.push(run.env.observe());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(|error| persistence_error(error).0)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"run_completed","snapshot":snapshot.clone()}).to_string(),
    ));
    Ok(Json(snapshot))
}
async fn record_decision(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(input): Json<DecisionInput>,
) -> Result<Json<DecisionRecord>, (StatusCode, String)> {
    if input.decision_summary.chars().count() > 500 {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "decision_summary must be at most 500 characters".into(),
        ));
    }
    if input.provider.chars().count() > 128
        || input
            .model
            .as_ref()
            .is_some_and(|model| model.chars().count() > 128)
    {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "provider and model must be at most 128 characters".into(),
        ));
    }
    if let Some(metadata) = &input.agent_metadata {
        let finite = [
            metadata.confidence,
            metadata.value_estimate,
            metadata.policy_entropy,
        ]
        .into_iter()
        .flatten()
        .all(f64::is_finite);
        let confidence_valid = metadata
            .confidence
            .is_none_or(|value| (0.0..=1.0).contains(&value));
        let entropy_valid = metadata.policy_entropy.is_none_or(|value| value >= 0.0);
        let planner_valid = metadata.planner.as_ref().is_none_or(|planner| {
            !planner.name.is_empty()
                && planner.name.chars().count() <= 128
                && planner
                    .planning_time_ms
                    .is_none_or(|value| value.is_finite() && value >= 0.0)
        });
        if !finite || !confidence_valid || !entropy_valid || !planner_valid {
            return Err((
                StatusCode::UNPROCESSABLE_ENTITY,
                "agent_metadata contains invalid bounded telemetry".into(),
            ));
        }
    }
    let mut runs = state.runs.lock().await;
    let run = runs
        .get_mut(&id)
        .ok_or((StatusCode::NOT_FOUND, "run not found".into()))?;
    if run.env.done {
        return Err((StatusCode::CONFLICT, "run is terminal".into()));
    }
    let decision = DecisionRecord {
        step: run.env.step.saturating_add(1),
        action: input.action.0,
        decision_summary: input.decision_summary,
        provider: input.provider,
        model: input.model,
        latency_ms: input.latency_ms,
        token_usage: input.token_usage,
        agent_metadata: input.agent_metadata,
    };
    run.decisions.push(decision.clone());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(persistence_error)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"agent_decision","decision":decision}).to_string(),
    ));
    Ok(Json(decision))
}
async fn step(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(StrictAction(action)): Json<StrictAction>,
) -> Result<Json<StepResult>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if run.paused || run.env.done {
        return Err(StatusCode::CONFLICT);
    };
    let control_started = Instant::now();
    run.actions.push(action.clone());
    let result = run.env.step(action);
    run.control
        .simulation_latency_us
        .push(control_started.elapsed().as_micros() as u64);
    match run.controller {
        Controller::Provider => run.control.provider_steps += 1,
        Controller::Manual => run.control.manual_steps += 1,
    }
    if run.env.done {
        run.transition_control(ControlPhase::Finished);
    }
    run.history.push(run.env.snapshot());
    run.observations.push(result.observation.clone());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(|error| persistence_error(error).0)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"simulation_event","result":result,"snapshot":run.env.snapshot()}).to_string(),
    ));
    Ok(Json(result))
}
async fn manual_step(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    Json(StrictAction(action)): Json<StrictAction>,
) -> Result<Json<StepResult>, StatusCode> {
    let mut runs = state.runs.lock().await;
    let run = runs.get_mut(&id).ok_or(StatusCode::NOT_FOUND)?;
    if !run.paused || run.env.done {
        return Err(StatusCode::CONFLICT);
    }
    run.transition_control(ControlPhase::Manual);
    let control_started = Instant::now();
    run.actions.push(action.clone());
    let result = run.env.step(action);
    run.control
        .simulation_latency_us
        .push(control_started.elapsed().as_micros() as u64);
    if run.env.done {
        run.transition_control(ControlPhase::Finished);
    }
    run.control.manual_steps += 1;
    run.history.push(run.env.snapshot());
    run.observations.push(result.observation.clone());
    persist(
        &run.env,
        run.world_manifest.as_ref(),
        &run.history,
        &run.observations,
        &run.decisions,
        &run.actions,
        run.control_snapshot(),
    )
    .map_err(|error| persistence_error(error).0)?;
    let _ = state.updates.send((
        id,
        serde_json::json!({"type":"manual_simulation_event","result":result,"snapshot":run.env.snapshot()}).to_string(),
    ));
    Ok(Json(result))
}
async fn ws(
    Path(id): Path<Uuid>,
    State(state): State<AppState>,
    socket: WebSocketUpgrade,
) -> Result<axum::response::Response, StatusCode> {
    let initial = state
        .runs
        .lock()
        .await
        .get(&id)
        .map(|r| serde_json::json!({"type":"state","snapshot":r.env.snapshot(),"observation":r.env.observe()}).to_string())
        .ok_or(StatusCode::NOT_FOUND)?;
    Ok(socket.on_upgrade(move |socket| ws_session(socket, id, state.updates.subscribe(), initial)))
}
async fn ws_session(
    mut socket: WebSocket,
    id: Uuid,
    mut updates: broadcast::Receiver<(Uuid, String)>,
    initial: String,
) {
    if socket.send(Message::Text(initial.into())).await.is_err() {
        return;
    }
    loop {
        tokio::select! {
            update = updates.recv() => match update { Ok((run_id, message)) if run_id == id => if socket.send(Message::Text(message.into())).await.is_err() { break }, Ok(_) => {}, Err(_) => break },
            inbound = socket.next() => if inbound.is_none() { break },
        }
    }
}

fn app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/api/scenarios", get(scenarios))
        .route("/api/scenarios/{id}", get(scenario_by_id))
        .route("/api/worlds/generate", post(generate_world))
        .route("/api/worlds/partition/{seed}", get(world_partition))
        .route("/api/runs", get(runs).post(create))
        .route("/api/runs/{id}", get(snapshot))
        .route("/api/runs/{id}/status", get(run_status))
        .route("/api/runs/{id}/observation", get(observation))
        .route("/api/runs/{id}/step", post(step))
        .route("/api/runs/{id}/manual-step", post(manual_step))
        .route("/api/runs/{id}/pause", post(pause))
        .route("/api/runs/{id}/resume", post(resume))
        .route("/api/runs/{id}/abort", post(abort))
        .route("/api/runs/{id}/provider-error", post(provider_error))
        .route("/api/runs/{id}/stop", post(stop_run))
        .route("/api/runs/{id}/decision", post(record_decision))
        .route("/api/runs/{id}/events", get(events))
        .route("/api/runs/{id}/replay", get(replay))
        .route("/api/replays", get(archived_replays))
        .route("/api/replays/{id}/resume", post(restore_replay))
        .route("/api/benchmarks", get(benchmarks))
        .route("/ws/runs/{id}", get(ws))
        .fallback_service(ServeDir::new("../frontend"))
        .layer(DefaultBodyLimit::max(64 * 1024))
        .layer(
            CorsLayer::new()
                .allow_origin([
                    "http://localhost:3000".parse().unwrap(),
                    "http://127.0.0.1:3000".parse().unwrap(),
                ])
                .allow_methods([Method::GET, Method::POST])
                .allow_headers([CONTENT_TYPE]),
        )
        .with_state(state)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_some_and(|value| value == "--validate-scenario")
    {
        let path = arguments
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("--validate-scenario requires a JSON path"))?;
        if arguments.len() != 2 {
            anyhow::bail!("usage: sim-server --validate-scenario <scenario.json>");
        }
        let raw = std::fs::read(path)?;
        let configured: Scenario = serde_json::from_slice(&raw)?;
        configured.validate()?;
        println!(
            "valid scenario: {} v{} ({}x{}, {} exits, {} routes)",
            configured.id,
            configured.version,
            configured.width,
            configured.height,
            configured.doors.iter().filter(|door| door.is_exit).count(),
            configured.routes.len()
        );
        return Ok(());
    }
    let log_filter = std::env::var("RUST_LOG")
        .or_else(|_| std::env::var("LOG_LEVEL"))
        .unwrap_or_else(|_| "info".into());
    tracing_subscriber::fmt().with_env_filter(log_filter).init();
    let app = app(AppState::default());
    let port = std::env::var("SIM_SERVER_PORT")
        .ok()
        .and_then(|value| value.parse::<u16>().ok())
        .unwrap_or(8080);
    let address = format!("127.0.0.1:{port}");
    let listener = tokio::net::TcpListener::bind(&address).await?;
    tracing::info!(%address, "Rust simulation server listening");
    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use sim_core::Direction;
    use tower::ServiceExt;

    #[test]
    fn persistence_reports_an_unusable_replay_directory() {
        let env = Environment::new(bundled_scenario(), 1).unwrap();
        let path = std::env::temp_dir().join(format!("sim-server-persist-{}", Uuid::new_v4()));
        std::fs::write(&path, "not a directory").unwrap();
        let error = persist_to(
            &path,
            PersistenceInput {
                env: &env,
                world_manifest: None,
                timeline: &[],
                observations: &[],
                decisions: &[],
                actions: &[],
                control: ControlStats::default(),
            },
        )
        .unwrap_err();
        std::fs::remove_file(path).unwrap();
        assert!(error.contains("cannot create replay directory"));
    }

    #[test]
    fn committed_schema_vocabulary_matches_rust_action_boundary() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../../schemas/action-request.v1.json"))
                .unwrap();
        let examples = [
            serde_json::json!({"type":"move","direction":"north"}),
            serde_json::json!({"type":"inspect"}),
            serde_json::json!({"type":"pickup"}),
            serde_json::json!({"type":"drop","item_id":"item"}),
            serde_json::json!({"type":"use_item","item_id":"item"}),
            serde_json::json!({"type":"open","target_id":"door"}),
            serde_json::json!({"type":"close","target_id":"door"}),
            serde_json::json!({"type":"talk","target_id":"npc"}),
            serde_json::json!({"type":"give","target_id":"npc","item_id":"item"}),
            serde_json::json!({"type":"wait"}),
            serde_json::json!({"type":"rest"}),
        ];
        let vocabulary = schema["properties"]["type"]["enum"].as_array().unwrap();
        assert_eq!(vocabulary.len(), examples.len());
        for example in examples {
            assert!(vocabulary.contains(&example["type"]));
            assert!(serde_json::from_value::<StrictAction>(example.clone()).is_ok());
            let mut invalid = example;
            invalid["unknown_field"] = serde_json::json!(true);
            assert!(serde_json::from_value::<StrictAction>(invalid).is_err());
        }
        for direction in schema["properties"]["direction"]["anyOf"][0]["enum"]
            .as_array()
            .unwrap()
        {
            assert!(
                serde_json::from_value::<StrictAction>(
                    serde_json::json!({"type":"move","direction":direction})
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn committed_decision_schema_matches_rust_metadata_boundary() {
        let schema: serde_json::Value =
            serde_json::from_str(include_str!("../../../../schemas/agent-decision.v1.json"))
                .unwrap();
        let metadata = &schema["properties"]["agent_metadata"]["anyOf"][0];
        assert_eq!(schema["required"], serde_json::json!(["action"]));
        assert_eq!(metadata["properties"]["confidence"]["maximum"], 1);
        assert_eq!(
            metadata["properties"]["planner"]["properties"]["name"]["maxLength"],
            128
        );
        let accepted = serde_json::json!({
            "action": {"type": "wait"}, "decision_summary": "bounded",
            "provider": "test", "model": null, "latency_ms": 1, "token_usage": null,
            "agent_metadata": {"confidence": 0.5, "value_estimate": -2.0, "policy_entropy": 0.1, "planner": {"name": "astar", "expanded_nodes": 4, "planning_time_ms": 1.5}}
        });
        assert!(serde_json::from_value::<DecisionInput>(accepted).is_ok());
        let rejected = serde_json::json!({
            "action": {"type": "wait"}, "decision_summary": "bounded",
            "provider": "test", "model": null, "latency_ms": 1, "token_usage": null,
            "agent_metadata": {"confidence": 0.5, "reasoning": "not a contract field"}
        });
        assert!(serde_json::from_value::<DecisionInput>(rejected).is_err());
    }

    #[test]
    fn bundled_survival_room_script_escapes() {
        let mut env = Environment::new(scenario(), 42).unwrap();
        let moves = |direction, count| {
            std::iter::repeat_n(Action::Move { direction }, count).collect::<Vec<_>>()
        };
        let mut actions = moves(Direction::East, 3);
        actions.extend(moves(Direction::North, 2));
        actions.push(Action::Open {
            target_id: "key_locker".into(),
        });
        actions.push(Action::Pickup { item_id: None });
        actions.extend(moves(Direction::South, 3));
        actions.extend(moves(Direction::East, 2));
        actions.push(Action::Open {
            target_id: "service_door".into(),
        });
        actions.extend(moves(Direction::East, 3));
        actions.extend(moves(Direction::South, 1));
        actions.extend(moves(Direction::East, 4));
        actions.extend(moves(Direction::North, 2));
        actions.push(Action::Open {
            target_id: "exit_door".into(),
        });
        actions.extend(moves(Direction::East, 1));
        for action in actions {
            env.step(action);
        }
        assert!(env.done);
        assert_eq!(env.terminal_reason.as_deref(), Some("escaped"));
        assert_eq!(env.step, 25);
    }

    #[test]
    fn npc_guided_survival_room_strategy_also_escapes_without_hazard_damage() {
        let mut env = Environment::new(scenario(), 42).unwrap();
        let moves = |direction, count| {
            std::iter::repeat_n(Action::Move { direction }, count).collect::<Vec<_>>()
        };
        let mut actions = moves(Direction::East, 2);
        actions.extend(moves(Direction::South, 1));
        actions.push(Action::Talk {
            target_id: "caretaker".into(),
            message: Some("How do I leave?".into()),
        });
        actions.extend(moves(Direction::East, 1));
        actions.extend(moves(Direction::North, 3));
        actions.push(Action::Open {
            target_id: "key_locker".into(),
        });
        actions.push(Action::Pickup { item_id: None });
        actions.extend(moves(Direction::South, 3));
        actions.extend(moves(Direction::East, 2));
        actions.push(Action::Open {
            target_id: "service_door".into(),
        });
        actions.extend(moves(Direction::East, 3));
        actions.extend(moves(Direction::South, 1));
        actions.extend(moves(Direction::East, 4));
        actions.extend(moves(Direction::North, 2));
        actions.push(Action::Open {
            target_id: "exit_door".into(),
        });
        actions.extend(moves(Direction::East, 1));
        for action in actions {
            env.step(action);
        }
        assert_eq!(env.terminal_reason.as_deref(), Some("escaped"));
        assert_eq!(env.metrics().npc_interactions, 1);
        assert_eq!(env.metrics().hazard_damage_taken, 0);
        assert!(env.events.iter().any(|event| event.kind == "NpcSpoke"));
    }

    #[tokio::test]
    async fn control_clock_accounts_for_wall_time_and_freezes_at_completion() {
        use std::time::Duration;
        let state = AppState::default();
        let (_, Json(created)) = create(State(state.clone()), None).await.unwrap();
        let id: Uuid = created["run_id"].as_str().unwrap().parse().unwrap();
        let mut runs = state.runs.lock().await;
        let run = runs.get_mut(&id).unwrap();
        let start = run.control_since;
        run.transition_control_at(ControlPhase::Paused, start + Duration::from_millis(100));
        run.transition_control_at(ControlPhase::Manual, start + Duration::from_millis(300));
        run.transition_control_at(ControlPhase::Provider, start + Duration::from_millis(600));
        run.transition_control_at(ControlPhase::Finished, start + Duration::from_millis(1000));
        let totals = run.control_snapshot_at(start + Duration::from_secs(60));
        assert_eq!(totals.provider_control_ms, 500);
        assert_eq!(totals.paused_ms, 200);
        assert_eq!(totals.manual_control_ms, 300);
    }

    #[tokio::test]
    async fn manual_runs_count_steps_correctly_and_terminal_controls_are_idempotent() {
        let state = AppState::default();
        let request =
            serde_json::from_value(serde_json::json!({"controller":"manual","max_steps":1}))
                .unwrap();
        let (_, Json(created)) = create(State(state.clone()), Some(Json(request)))
            .await
            .unwrap();
        let id: Uuid = created["run_id"].as_str().unwrap().parse().unwrap();
        let _ = step(
            Path(id),
            State(state.clone()),
            Json(StrictAction(Action::Wait)),
        )
        .await
        .unwrap();
        let before = {
            let runs = state.runs.lock().await;
            let run = runs.get(&id).unwrap();
            let totals = run.control_snapshot();
            assert_eq!(totals.manual_steps, 1);
            assert_eq!(totals.provider_steps, 0);
            assert_eq!(totals.provider_control_ms, 0);
            assert_eq!(totals.simulation_latency_us.len(), 1);
            (run.history.len(), serde_json::to_value(totals).unwrap())
        };
        assert_eq!(
            pause(Path(id), State(state.clone())).await.unwrap_err(),
            StatusCode::CONFLICT
        );
        assert_eq!(
            resume(Path(id), State(state.clone())).await.unwrap_err(),
            StatusCode::CONFLICT
        );
        let _ = abort(Path(id), State(state.clone())).await.unwrap();
        let _ = provider_error(Path(id), State(state.clone()))
            .await
            .unwrap();
        let runs = state.runs.lock().await;
        let run = runs.get(&id).unwrap();
        assert_eq!(run.history.len(), before.0);
        assert_eq!(
            serde_json::to_value(run.control_snapshot()).unwrap(),
            before.1
        );
    }

    #[tokio::test]
    async fn step_broadcasts_include_current_researcher_snapshot() {
        let state = AppState::default();
        let (_, Json(created)) = create(State(state.clone()), None).await.unwrap();
        let id: Uuid = created["run_id"].as_str().unwrap().parse().unwrap();
        let mut updates = state.updates.subscribe();
        let _ = step(
            Path(id),
            State(state.clone()),
            Json(StrictAction(Action::Wait)),
        )
        .await
        .unwrap();
        let (_, message) = updates.try_recv().unwrap();
        let update: serde_json::Value = serde_json::from_str(&message).unwrap();
        assert_eq!(update["type"], "simulation_event");
        assert_eq!(update["snapshot"]["step"], 1);
        assert_eq!(update["result"]["observation"]["step"], 1);
        assert_eq!(update["snapshot"]["run_id"], created["run_id"]);
        let _ = pause(Path(id), State(state.clone())).await.unwrap();
        updates.try_recv().unwrap();
        let _ = manual_step(Path(id), State(state), Json(StrictAction(Action::Wait)))
            .await
            .unwrap();
        let (_, message) = updates.try_recv().unwrap();
        let update: serde_json::Value = serde_json::from_str(&message).unwrap();
        assert_eq!(update["type"], "manual_simulation_event");
        assert_eq!(update["snapshot"]["step"], 2);
        assert_eq!(update["result"]["observation"]["step"], 2);
    }

    #[tokio::test]
    async fn paused_run_rejects_steps() {
        let app = app(AppState::default());
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();

        let replay = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(replay.status(), StatusCode::OK);
        let replay_body = axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap();
        let replay_json = serde_json::from_slice::<serde_json::Value>(&replay_body).unwrap();
        assert_eq!(replay_json["seed"], 42);
        assert_eq!(replay_json["scenario_id"], "survival_room");
        assert_eq!(replay_json["scenario_version"], 5);
        assert_eq!(replay_json["observation_mode"], "normal");
        assert_eq!(replay_json["timeline"].as_array().unwrap().len(), 1);
        assert_eq!(replay_json["observations"].as_array().unwrap().len(), 1);

        let pause = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/pause"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(pause.status(), StatusCode::OK);

        let status = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/status"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        let status_body = axum::body::to_bytes(status.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&status_body).unwrap()["paused"],
            true
        );

        let step = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(step.status(), StatusCode::CONFLICT);

        let manual_step = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/manual-step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(manual_step.status(), StatusCode::OK);
        let manual_body = axum::body::to_bytes(manual_step.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&manual_body).unwrap()["step_number"],
            1
        );

        let replay_after_step = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_after_step_body =
            axum::body::to_bytes(replay_after_step.into_body(), usize::MAX)
                .await
                .unwrap();
        let replay_after_step_json =
            serde_json::from_slice::<serde_json::Value>(&replay_after_step_body).unwrap();
        assert_eq!(
            replay_after_step_json["timeline"].as_array().unwrap().len(),
            2
        );
        assert_eq!(
            replay_after_step_json["observations"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            replay_after_step_json["actions"].as_array().unwrap().len(),
            1
        );
        assert_eq!(replay_after_step_json["control"]["manual_steps"], 1);
        assert_eq!(replay_after_step_json["control"]["provider_steps"], 0);
        assert!(replay_after_step_json["control"]["paused_ms"].is_u64());

        let still_paused = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(still_paused.status(), StatusCode::CONFLICT);
    }

    #[tokio::test]
    async fn scenario_lookup_and_benchmark_summary_are_available() {
        let app = app(AppState::default());
        let catalog = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/scenarios")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(catalog.status(), StatusCode::OK);
        let catalog_body = axum::body::to_bytes(catalog.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(
            serde_json::from_slice::<serde_json::Value>(&catalog_body)
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .any(|scenario| scenario["id"] == "survival_room")
        );
        let known = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/scenarios/survival_room")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(known.status(), StatusCode::OK);
        let known_body = axum::body::to_bytes(known.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&known_body).unwrap()["id"],
            "survival_room"
        );

        let missing = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/scenarios/unknown")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(missing.status(), StatusCode::NOT_FOUND);

        let selected = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"scenario_id":"survival_room","seed":7}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(selected.status(), StatusCode::CREATED);

        let unknown_selection = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"scenario_id":"missing"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(unknown_selection.status(), StatusCode::BAD_REQUEST);

        let generated_run = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":7,"generated_world":{"seed":99}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generated_run.status(), StatusCode::CREATED);
        let generated_run_body = axum::body::to_bytes(generated_run.into_body(), usize::MAX)
            .await
            .unwrap();
        let generated_run_json =
            serde_json::from_slice::<serde_json::Value>(&generated_run_body).unwrap();
        assert_eq!(generated_run_json["world_manifest"]["seed"], 99);
        assert!(
            generated_run_json["world_manifest"]["validation"]["solvable"]
                .as_bool()
                .unwrap()
        );

        let held_out_generated_run = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"seed":7,"generated_world":{"seed":9000,"partition":"test"}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(held_out_generated_run.status(), StatusCode::CREATED);

        let mismatched_partition = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/worlds/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42,"partition":"test"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(mismatched_partition.status(), StatusCode::BAD_REQUEST);

        let generated = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/worlds/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(generated.status(), StatusCode::OK);
        let generated_body = axum::body::to_bytes(generated.into_body(), usize::MAX)
            .await
            .unwrap();
        let generated_json = serde_json::from_slice::<serde_json::Value>(&generated_body).unwrap();
        assert!(generated_json["validation"]["solvable"].as_bool().unwrap());
        assert!(
            generated_json["world_hash"]
                .as_str()
                .unwrap()
                .starts_with("fnv1a64:")
        );

        let held_out = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/worlds/partition/9000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(held_out.status(), StatusCode::OK);
        let held_out_body = axum::body::to_bytes(held_out.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&held_out_body).unwrap(),
            "test"
        );
        let outside_distribution = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/worlds/partition/10000")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(outside_distribution.status(), StatusCode::NOT_FOUND);

        let benchmark = app
            .oneshot(
                Request::builder()
                    .uri("/api/benchmarks")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(benchmark.status(), StatusCode::OK);
        let benchmark_body = axum::body::to_bytes(benchmark.into_body(), usize::MAX)
            .await
            .unwrap();
        let benchmark_json = serde_json::from_slice::<serde_json::Value>(&benchmark_body).unwrap();
        assert!(benchmark_json["total_runs"].is_u64());
        assert!(benchmark_json["mean_normalized_score"].is_number());
        assert!(benchmark_json["total_provider_calls"].is_u64());
        assert!(benchmark_json["mean_provider_latency_ms"].is_null());
    }

    #[tokio::test]
    async fn max_step_override_is_validated_by_the_authoritative_server() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42,"max_steps":1}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let stepped = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let result = axum::body::to_bytes(stepped.into_body(), usize::MAX)
            .await
            .unwrap();
        let json = serde_json::from_slice::<serde_json::Value>(&result).unwrap();
        assert_eq!(json["terminal_reason"], "timeout");
        assert_eq!(json["step_number"], 1);
        let rejected = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status(), StatusCode::CONFLICT);
        let replay_response = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_body = axum::body::to_bytes(replay_response.into_body(), usize::MAX)
            .await
            .unwrap();
        let replay_json: serde_json::Value = serde_json::from_slice(&replay_body).unwrap();
        assert_eq!(replay_json["actions"].as_array().unwrap().len(), 1);
        assert_eq!(replay_json["timeline"].as_array().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn observation_mode_is_applied_and_recorded_in_replay_metadata() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42,"observation_mode":"minimal"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let created_json = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        assert_eq!(created_json["observation"]["observation_mode"], "minimal");
        let run_id = created_json["run_id"].as_str().unwrap();
        let replay = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_body = axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&replay_body).unwrap()["observation_mode"],
            "minimal"
        );
    }

    #[tokio::test]
    async fn custom_reward_configuration_is_authoritative_and_replayable() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"seed":42,"reward_config":{"baseline_per_step":-3,"discovery_bonus":7,"invalid_action_penalty":-4,"terminal_success":50,"terminal_failure":-60}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(created.status(), StatusCode::CREATED);
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let created_json = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        assert_eq!(created_json["reward_config"]["baseline_per_step"], -3);
        let run_id = created_json["run_id"].as_str().unwrap();
        let replay = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_body = axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap();
        let replay_json = serde_json::from_slice::<serde_json::Value>(&replay_body).unwrap();
        assert_eq!(replay_json["reward_config"]["terminal_success"], 50);
    }

    #[tokio::test]
    async fn provider_decisions_are_validated_and_persisted_in_replays() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let decision = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/decision"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"action":{"type":"wait"},"decision_summary":"Assess the room.","provider":"scripted","model":null,"latency_ms":0,"token_usage":{"input_tokens":12,"output_tokens":7,"cached_tokens":null,"total_tokens":19},"agent_metadata":{"confidence":0.8,"value_estimate":1.5,"policy_entropy":0.2,"planner":{"name":"astar","expanded_nodes":12,"planning_time_ms":3.5}}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(decision.status(), StatusCode::OK);
        let replay = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_body = axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap();
        let replay_json = serde_json::from_slice::<serde_json::Value>(&replay_body).unwrap();
        assert_eq!(replay_json["decisions"].as_array().unwrap().len(), 1);
        assert_eq!(
            replay_json["decisions"][0]["decision_summary"],
            "Assess the room."
        );
        assert_eq!(
            replay_json["decisions"][0]["token_usage"]["total_tokens"],
            19
        );
        assert_eq!(
            replay_json["decisions"][0]["agent_metadata"]["planner"]["name"],
            "astar"
        );
    }

    #[tokio::test]
    async fn unknown_action_fields_are_rejected_at_the_http_boundary() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let invalid = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait","unexpected":true}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid.status(), StatusCode::UNPROCESSABLE_ENTITY);

        let oversized_target = "x".repeat(129);
        let invalid_length = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        serde_json::json!({
                            "type": "open",
                            "target_id": oversized_target,
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(invalid_length.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn unknown_generator_config_fields_are_rejected_at_the_http_boundary() {
        let response = app(AppState::default())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/worlds/generate")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"seed":42,"config":{"generator_version":1,"min_width":9,"max_width":11,"min_height":7,"max_height":9,"max_attempts":16,"min_rooms":2,"max_rooms":3,"hazard_kinds":["electrical"],"min_wdith":9}}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn provider_error_persists_a_terminal_partial_run() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":42}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
            .as_str()
            .unwrap()
            .to_owned();
        let terminal = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/provider-error"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let terminal_body = axum::body::to_bytes(terminal.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&terminal_body).unwrap()["terminal_reason"],
            "provider_error"
        );
        let replay = app
            .oneshot(
                Request::builder()
                    .uri(format!("/api/runs/{run_id}/replay"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let replay_body = axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap();
        assert!(
            serde_json::from_slice::<serde_json::Value>(&replay_body).unwrap()["events"]
                .as_array()
                .unwrap()
                .iter()
                .any(|event| event["type"] == "RunInterrupted")
        );
    }

    #[tokio::test]
    async fn orchestration_budget_stop_reasons_are_strict_and_authoritative() {
        for reason in ["client_timeout", "token_budget_exhausted"] {
            let app = app(AppState::default());
            let created = app
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/api/runs")
                        .header("content-type", "application/json")
                        .body(Body::from(r#"{"seed":42}"#))
                        .unwrap(),
                )
                .await
                .unwrap();
            let body = axum::body::to_bytes(created.into_body(), usize::MAX)
                .await
                .unwrap();
            let run_id = serde_json::from_slice::<serde_json::Value>(&body).unwrap()["run_id"]
                .as_str()
                .unwrap()
                .to_owned();
            let stopped = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(format!("/api/runs/{run_id}/stop"))
                        .header("content-type", "application/json")
                        .body(Body::from(format!(r#"{{"reason":"{reason}"}}"#)))
                        .unwrap(),
                )
                .await
                .unwrap();
            let stopped_body = axum::body::to_bytes(stopped.into_body(), usize::MAX)
                .await
                .unwrap();
            assert_eq!(
                serde_json::from_slice::<serde_json::Value>(&stopped_body).unwrap()["terminal_reason"],
                reason
            );
        }
    }

    #[tokio::test]
    async fn generated_world_replay_can_reconstruct_without_catalog_state() {
        let app = app(AppState::default());
        let created = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/runs")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"seed":987,"generated_world":{"seed":654}}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = axum::body::to_bytes(created.into_body(), usize::MAX)
            .await
            .unwrap();
        let created_json = serde_json::from_slice::<serde_json::Value>(&body).unwrap();
        assert_eq!(created_json["world_manifest"]["seed"], 654);
        let run_id = created_json["run_id"].as_str().unwrap().to_owned();
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/runs/{run_id}/step"))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"type":"wait"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        let restored = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/replays/{run_id}/resume"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let restored_status = restored.status();
        let restored_body = axum::body::to_bytes(restored.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(
            restored_status,
            StatusCode::CREATED,
            "{}",
            String::from_utf8_lossy(&restored_body)
        );
        let restored_json = serde_json::from_slice::<serde_json::Value>(&restored_body).unwrap();
        assert_eq!(restored_json["restored_from"], run_id);
        assert_eq!(restored_json["snapshot"]["step"], 1);
        assert_ne!(restored_json["run_id"], run_id);
    }
}
