"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Pos = { x: number; y: number };
type Snapshot = { run_id: string; step: number; goal: string; width: number; height: number; walls: Pos[]; agent: { position: Pos; facing: string; health: number; energy: number; hydration: number; inventory: string[]; status_effects: string[]; alive: boolean; escaped: boolean }; items: { id: string; position: Pos }[]; hazards: { id?: string; position: Pos; kind?: string; active?: boolean }[]; doors: { id?: string; position: Pos }[]; containers: { id?: string; position: Pos }[]; npc?: { position: Pos; disposition?: string; trust?: number }; done: boolean; terminal_reason?: string; metrics: { normalized_score: number; invalid_actions: number; unique_cells_visited: number; hazard_damage_taken: number; simulated_time: number; unnecessary_actions: number; recovery_after_failure: boolean; discovered_doors?: number; discovered_items?: number; discovered_hazards?: number; discovered_npcs?: number; exploration_coverage?: number; action_diversity?: number; resource_efficiency?: number; carried_weight?: number; inventory_weight_capacity?: number } };
type Observation = { step: number; observation_mode: string; visible_cells: { relative_position: Pos; terrain: string; entities: { id: string; type: string; state?: string; name?: string }[] }[]; recent_events: string[]; perception_note?: string };
type DecisionRecord = { step: number; action: { type: string; direction?: string; target_id?: string; item_id?: string }; decision_summary: string; provider: string; model?: string; latency_ms?: number };
type RunEvent = { event_id?: string; type?: string; step?: number; message: string };
const notice = (message: string): RunEvent => ({ type: "observer", message });
function mergeEvents(incoming: RunEvent[], existing: RunEvent[]): RunEvent[] {
  const seen = new Set<string>();
  return [...incoming].reverse().concat(existing).filter((event) => {
    if (!event.event_id) return true;
    if (seen.has(event.event_id)) return false;
    seen.add(event.event_id);
    return true;
  }).slice(0, 1000);
}
type WorldManifest = { seed: number; world_hash: string; generator_version: number; generation_attempt: number; dimensions: Pos; validation: { solvable: boolean } };
type RewardConfig = { baseline_per_step: number; discovery_bonus: number; invalid_action_penalty: number; terminal_success: number; terminal_failure: number };
type Replay = { snapshot: Snapshot; events: RunEvent[]; timeline?: Snapshot[]; observations?: Observation[]; decisions?: DecisionRecord[]; world_manifest?: WorldManifest | null; reward_config?: RewardConfig };
type ScenarioSummary = { id: string; name: string; version: number; goal: string; max_steps: number; time_limit?: number; width: number; height: number };
const API = process.env.NEXT_PUBLIC_SIM_SERVER_URL ?? "http://localhost:8080";
const WS = (process.env.NEXT_PUBLIC_SIM_WS_URL ?? "ws://localhost:8080").replace(/\/$/, "");
const AGENT_API = process.env.NEXT_PUBLIC_AGENT_SERVICE_URL ?? "http://127.0.0.1:8090";

export default function Observer({ initialRunId }: { initialRunId?: string }) {
  const [run, setRun] = useState<string>();
  const [state, setState] = useState<Snapshot>();
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [paused, setPaused] = useState(false);
  const [runs, setRuns] = useState<Snapshot[]>([]);
  const [error, setError] = useState<string>();
  const [lastAction, setLastAction] = useState("No action submitted.");
  const [frames, setFrames] = useState<Snapshot[]>([]);
  const [observationFrames, setObservationFrames] = useState<Observation[]>([]);
  const [agentObservation, setAgentObservation] = useState<Observation>();
  const [agentDecision, setAgentDecision] = useState<DecisionRecord>();
  const [decisionFrames, setDecisionFrames] = useState<DecisionRecord[]>([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [observationMode, setObservationMode] = useState("normal");
  const [provider, setProvider] = useState("scripted");
  const [agentRunStatus, setAgentRunStatus] = useState<string>();
  const [agentExports, setAgentExports] = useState<{ jsonl?: string; parquet?: string; experiment_manifest?: string }>({});
  const [controlMode, setControlMode] = useState<"manual" | "agent" | "replay">();
  const [seed, setSeed] = useState(42);
  const [maxSteps, setMaxSteps] = useState("60");
  const [memoryMode, setMemoryMode] = useState("recent");
  const [memoryWindow, setMemoryWindow] = useState(5);
  const [model, setModel] = useState("");
  const [eventFilter, setEventFilter] = useState("");
  const [eventType, setEventType] = useState("all");
  const [replaySearch, setReplaySearch] = useState("");
  const [theme, setTheme] = useState<"dark" | "light">("dark");
  const [scenarioInfo, setScenarioInfo] = useState<ScenarioSummary>();
  const [serviceHealth, setServiceHealth] = useState({ rust: false, agent: false });
  const [connection, setConnection] = useState<"idle" | "connected" | "reconnecting" | "offline">("idle");
  const [researchMetadata, setResearchMetadata] = useState<{ worldManifest?: WorldManifest | null; rewardConfig?: RewardConfig }>({});
  const reconnectAttempt = useRef(0);
  const decisionForStep = useCallback((step: number, decisions = decisionFrames) => decisions.filter((decision) => decision.step <= step).at(-1), [decisionFrames]);

  async function refreshRuns() {
    try {
      const [active, archived, scenarios] = await Promise.all([fetch(API + "/api/runs"), fetch(API + "/api/replays"), fetch(API + "/api/scenarios")]);
      if (!active.ok || !archived.ok) throw new Error("Simulation server is unavailable.");
      const snapshots = (await active.json()) as Snapshot[];
      const replays = (await archived.json()) as Replay[];
      if (scenarios.ok) {
        const available = await scenarios.json() as ScenarioSummary[];
        if (available[0]?.id) setScenarioInfo(available[0]);
      }
      const unique = new Map(snapshots.map((snapshot) => [snapshot.run_id, snapshot]));
      replays.forEach((replay) => unique.set(replay.snapshot.run_id, replay.snapshot));
      setRuns([...unique.values()].sort((a, b) => b.step - a.step));
      setError(undefined);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "Unable to load runs."); }
  }
  useEffect(() => {
    const timer = window.setTimeout(() => { void refreshRuns(); }, 0);
    return () => window.clearTimeout(timer);
  }, []);
  useEffect(() => {
    if (!run || controlMode === "replay") return;
    let disposed = false;
    let socket: WebSocket | undefined;
    let retryTimer: number | undefined;
    const updateFromSocket = (data: string) => {
      try {
        const update = JSON.parse(data) as { type: string; snapshot?: Snapshot; observation?: Observation; decision?: DecisionRecord; result?: { events: RunEvent[]; observation?: Observation } };
        if (update.snapshot) setState(update.snapshot);
        if (update.observation) setAgentObservation(update.observation);
        if (update.decision) setAgentDecision(update.decision);
        if (update.result?.observation) setAgentObservation(update.result.observation);
        const resultEvents = update.result?.events;
        if (resultEvents) setEvents((old) => mergeEvents(resultEvents, old));
        if (update.type === "run_paused") setPaused(true);
        if (update.type === "run_resumed") setPaused(false);
      } catch {
        setError("Received an unreadable live update.");
      }
    };
    const recover = async () => {
      if (disposed) return;
      setConnection("reconnecting");
      try {
        const [snapshotResponse, replayResponse] = await Promise.all([
          fetch(API + "/api/runs/" + run),
          fetch(API + "/api/runs/" + run + "/replay"),
        ]);
        if (snapshotResponse.status === 404) {
          setConnection("offline");
          return;
        }
        if (!snapshotResponse.ok) throw new Error("State refresh failed");
        const snapshot = await snapshotResponse.json() as Snapshot;
        if (disposed) return;
        setState(snapshot);
        if (replayResponse.ok) {
          const replay = await replayResponse.json() as Replay;
          const observations = replay.observations ?? [];
          setAgentObservation(observations[observations.length - 1]);
          setResearchMetadata({ worldManifest: replay.world_manifest, rewardConfig: replay.reward_config });
        }
        connect();
      } catch {
        if (!disposed) scheduleReconnect();
      }
    };
    const scheduleReconnect = () => {
      if (disposed) return;
      const delay = Math.min(5000, 250 * 2 ** reconnectAttempt.current);
      reconnectAttempt.current += 1;
      retryTimer = window.setTimeout(() => { void recover(); }, delay);
    };
    const connect = () => {
      if (disposed) return;
      socket = new WebSocket(WS + "/ws/runs/" + run);
      socket.onopen = () => { reconnectAttempt.current = 0; setConnection("connected"); };
      socket.onmessage = ({ data }) => updateFromSocket(String(data));
      socket.onerror = () => socket?.close();
      socket.onclose = () => { if (!disposed) void recover(); };
    };
    connect();
    return () => { disposed = true; if (retryTimer !== undefined) window.clearTimeout(retryTimer); socket?.close(); };
  }, [run, controlMode]);
  useEffect(() => {
    if (!run || agentRunStatus !== "running") return;
    let disposed = false;
    const refreshAgentStatus = async () => {
      try {
        const response = await fetch(AGENT_API + "/api/agent-runs/" + run);
        if (!response.ok || disposed) return;
        const status = await response.json() as { status?: string; exports?: { jsonl?: string; parquet?: string; experiment_manifest?: string } };
        if (typeof status.status !== "string" || disposed) return;
        setAgentRunStatus(status.status);
        if (status.exports) setAgentExports(status.exports);
        if (status.status === "failed") setError("The provider worker failed; the authoritative replay remains available.");
      } catch {
        // The Rust WebSocket remains the live state authority if the optional control service restarts.
      }
    };
    void refreshAgentStatus();
    const timer = window.setInterval(() => { void refreshAgentStatus(); }, 1000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [run, agentRunStatus]);
  useEffect(() => {
    let disposed = false;
    const check = async () => {
      const [rust, agent] = await Promise.allSettled([fetch(API + "/health"), fetch(AGENT_API + "/health")]);
      if (!disposed) setServiceHealth({ rust: rust.status === "fulfilled" && rust.value.ok, agent: agent.status === "fulfilled" && agent.value.ok });
    };
    void check();
    const timer = window.setInterval(() => { void check(); }, 5000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, []);
  useEffect(() => {
    if (!playing || frameIndex >= frames.length - 1) return;
    const timer = window.setTimeout(() => {
      const next = frameIndex + 1;
      setFrameIndex(next);
      setState(frames[next]);
      if (observationFrames[next]) setAgentObservation(observationFrames[next]);
      setAgentDecision(decisionForStep(frames[next].step));
    }, 900 / speed);
    return () => window.clearTimeout(timer);
  }, [playing, frameIndex, frames, observationFrames, decisionForStep, speed]);

  async function startManual() {
    const response = await fetch(API + "/api/runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ controller: "manual", seed, max_steps: Number(maxSteps), observation_mode: observationMode }) });
    if (!response.ok) { setError("Unable to create a run."); return; }
    const data = await response.json() as { run_id: string; snapshot: Snapshot; observation?: Observation; world_manifest?: WorldManifest | null; reward_config?: RewardConfig };
    setRun(data.run_id); setConnection("reconnecting"); setState(data.snapshot); setAgentObservation(data.observation); setResearchMetadata({ worldManifest: data.world_manifest, rewardConfig: data.reward_config }); setAgentDecision(undefined); setFrames([]); setObservationFrames([]); setDecisionFrames([]); setPlaying(false); setEvents([notice("Manual run started.")]); setPaused(false); setAgentRunStatus(undefined); setAgentExports({}); setControlMode("manual"); setLastAction("Manual baseline initialized with seed " + seed + " / " + observationMode + " observation."); void refreshRuns();
  }
  async function startAgent() {
    const response = await fetch(AGENT_API + "/api/agent-runs", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ provider, seed, model: model || null, max_steps: Number(maxSteps), observation_mode: observationMode, memory_mode: memoryMode, memory_window: memoryWindow }) });
    if (!response.ok) { setError("Unable to start the local provider service. Start `python -m embodied_ai.agent_service` first."); return; }
    const data = await response.json() as { run_id: string; status: string };
    setRun(data.run_id); setConnection("reconnecting"); setState(undefined); setAgentObservation(undefined); setResearchMetadata({}); setAgentDecision(undefined); setFrames([]); setObservationFrames([]); setDecisionFrames([]); setPlaying(false); setEvents([notice("Provider run accepted: " + provider + ".")]); setPaused(false); setAgentRunStatus(data.status); setAgentExports({}); setControlMode("agent"); setLastAction("Waiting for " + provider + " to choose the first simulated action."); void refreshRuns();
  }
  const loadReplay = useCallback(async (id: string) => {
    const response = await fetch(API + "/api/runs/" + id + "/replay");
    if (!response.ok) { setError("Replay is no longer available."); return; }
    const replay = await response.json() as Replay;
    const timeline = replay.timeline?.length ? replay.timeline : [replay.snapshot];
    const replayObservations = replay.observations ?? [];
    const replayDecisions = replay.decisions ?? [];
    const lastFrame = timeline[timeline.length - 1];
    setRun(id); setConnection("idle"); setState(lastFrame); setAgentObservation(replayObservations[timeline.length - 1]); setResearchMetadata({ worldManifest: replay.world_manifest, rewardConfig: replay.reward_config }); setAgentDecision(replayDecisions.filter((decision) => decision.step <= lastFrame.step).at(-1)); setFrames(timeline); setObservationFrames(replayObservations); setDecisionFrames(replayDecisions); setFrameIndex(timeline.length - 1); setPlaying(false); setEvents([...replay.events].reverse()); setPaused(false); setAgentRunStatus(undefined); setAgentExports({}); setControlMode("replay"); setLastAction("Loaded persisted replay.");
  }, []);
  useEffect(() => {
    if (!initialRunId) return;
    const timer = window.setTimeout(() => { void loadReplay(initialRunId); }, 0);
    return () => window.clearTimeout(timer);
  }, [initialRunId, loadReplay]);
  async function act(type: string, direction?: string, target_id?: string, item_id?: string) {
    if (!run) return;
    if (controlMode === "agent" && !paused) { setEvents((old) => [notice("Pause the agent before taking a manual control step."), ...old]); return; }
    if (controlMode === "replay") { setEvents((old) => [notice("Loaded replays are read-only. Start a new manual run to control a body."), ...old]); return; }
    const endpoint = paused ? "/manual-step" : "/step";
    const response = await fetch(API + "/api/runs/" + run + endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ type, direction, target_id, item_id }) });
    if (!response.ok) { setEvents((old) => [notice("Action rejected: run is terminal or the control state changed."), ...old]); return; }
    const result = await response.json() as { events: RunEvent[]; observation?: Observation };
    const snapshot = await fetch(API + "/api/runs/" + run).then((item) => item.json()) as Snapshot;
    setState(snapshot); setAgentObservation(result.observation); setEvents((old) => mergeEvents(result.events, old)); setLastAction(type + (direction ? " " + direction : target_id ? " " + target_id : item_id ? " " + item_id : ""));
    if (snapshot.done) void refreshRuns();
  }
  async function control(op: "pause" | "resume" | "abort") {
    if (!run || controlMode === "replay" || state?.done) return;
    const response = await fetch(API + "/api/runs/" + run + "/" + op, { method: "POST" });
    if (!response.ok) { setError("Run control failed."); return; }
    if (op === "abort") { setState(await response.json() as Snapshot); void refreshRuns(); } else setPaused(op === "pause");
  }
  function selectFrame(next: number) {
    if (!frames[next]) return;
    setFrameIndex(next); setState(frames[next]); setAgentObservation(observationFrames[next]); setAgentDecision(decisionForStep(frames[next].step)); setPlaying(false);
  }
  const at = (position: Pos | undefined, x: number, y: number) => position?.x === x && position.y === y;
  const manualControlsEnabled = Boolean(run && controlMode !== "replay" && (controlMode !== "agent" || paused) && !state?.done);
  const width = state?.width ?? 15, height = state?.height ?? 9;
  const eventTypes = [...new Set(events.map((event) => event.type ?? "untyped").concat(eventType === "all" ? [] : [eventType]))].sort();
  const visibleEvents = events.filter((event) => (eventType === "all" || (event.type ?? "untyped") === eventType) && event.message.toLowerCase().includes(eventFilter.toLowerCase()));
  const visibleRuns = runs.filter((saved) => (saved.run_id + " " + (saved.terminal_reason ?? "active")).toLowerCase().includes(replaySearch.toLowerCase()));
  useEffect(() => {
    const keyboard = (event: KeyboardEvent) => {
      if (!manualControlsEnabled || event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
      const direction = ({ ArrowUp: "north", ArrowDown: "south", ArrowLeft: "west", ArrowRight: "east" } as Record<string, string>)[event.key];
      if (direction) { event.preventDefault(); void act("move", direction); }
      else if (event.key.toLowerCase() === "i") void act("inspect");
      else if (event.key.toLowerCase() === "w") void act("wait");
    };
    window.addEventListener("keydown", keyboard);
    return () => window.removeEventListener("keydown", keyboard);
  });
  return <main className={theme}>
    <header><div><p>RESEARCH OBSERVER / V1</p><h1>Embodied Worlds</h1></div><div className="launch-controls"><label>Seed <input aria-label="Seed" type="number" min="0" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></label><label>Max steps <input aria-label="Max steps" type="number" min="1" max="10000" value={maxSteps} onChange={(event) => setMaxSteps(event.target.value)} /></label><label>Observation <select aria-label="Observation mode" value={observationMode} onChange={(event) => setObservationMode(event.target.value)}><option value="minimal">Minimal</option><option value="normal">Local symbolic</option><option value="rich">Extended local</option><option value="noisy">Noisy local</option><option value="oracle">Oracle map (research)</option></select></label><label>Memory <select aria-label="Memory mode" value={memoryMode} onChange={(event) => setMemoryMode(event.target.value)}><option value="none">None</option><option value="recent">Recent</option></select></label><label>Memory window <input aria-label="Memory window" type="number" min="1" max="100" value={memoryWindow} disabled={memoryMode === "none"} onChange={(event) => setMemoryWindow(Number(event.target.value))} /></label><label>Provider <select aria-label="Provider" value={provider} onChange={(event) => setProvider(event.target.value)}><option value="scripted">Scripted</option><option value="mock_reasoning">Mock reasoning</option><option value="random_valid">Random valid</option><option value="cautious">Cautious</option><option value="explorer">Explorer</option><option value="gemini">Gemini</option></select></label><label>Model <input aria-label="Model" placeholder="environment default" value={model} onChange={(event) => setModel(event.target.value)} /></label><button onClick={startAgent}>Start agent</button><button onClick={startManual}>Start manual</button><button onClick={() => setTheme((value) => value === "dark" ? "light" : "dark")}>{theme === "dark" ? "Light" : "Dark"} mode</button></div></header>
    <section><div className="panel"><span>RESEARCHER VIEW</span><div className="map" style={{ gridTemplateColumns: "repeat(" + width + ",1fr)" }}>
      {Array.from({ length: width * height }, (_, index) => {
        const x = index % width, y = Math.floor(index / width);
        const hazard = state?.hazards.find((entity) => at(entity.position, x, y));
        const kind = at(state?.agent.position, x, y) ? "agent" : state?.walls.some((wall) => at(wall, x, y)) ? "wall" : hazard ? hazard.active === false ? "hazard inactive" : "hazard" : state?.doors.some((door) => at(door.position, x, y)) ? "door" : state?.containers.some((container) => at(container.position, x, y)) ? "container" : state?.items.some((item) => at(item.position, x, y) && !state.agent.inventory.includes(item.id)) ? "item" : at(state?.npc?.position, x, y) ? "npc" : "";
        return <div className={"cell " + kind} title={hazard?.kind ?? (kind || "floor")} key={index} />;
      })}</div><div className="legend"><i className="agent" />Agent <i className="item" />Item <i className="container" />Container <i className="door" />Door <i className="hazard" />Hazard <i className="npc" />NPC</div></div>
      <aside><div className="panel"><span>RUN STATE</span><h2>{state?.done ? state.terminal_reason ?? "completed" : paused ? "paused" : run ? "active" : "waiting"}</h2><p aria-live="polite">LIVE LINK: {connection}{agentRunStatus ? " · AGENT: " + agentRunStatus : ""}</p><p className="services">Rust {serviceHealth.rust ? "● online" : "○ offline"} · Agent {serviceHealth.agent ? "● online" : "○ offline"}</p><div className="metrics"><b>STEP {state?.step ?? 0}</b><b>HEALTH {state?.agent.health ?? 100}</b><b>ENERGY {state?.agent.energy ?? 100}</b><b>HYDRATION {state?.agent.hydration ?? 100}</b><b>FACING {state?.agent.facing ?? "east"}</b></div><button onClick={() => control(paused ? "resume" : "pause")} disabled={!run || controlMode === "replay" || state?.done}>{paused ? "Resume" : "Pause"}</button><button onClick={() => control("abort")} disabled={!run || controlMode === "replay" || state?.done}>Abort</button>{run && (agentExports.jsonl || agentExports.experiment_manifest) && <div className="downloads">{agentExports.jsonl && <a href={AGENT_API + "/api/agent-runs/" + run + "/exports/jsonl"}>Download JSONL</a>}{agentExports.parquet && <a href={AGENT_API + "/api/agent-runs/" + run + "/exports/parquet"}>Download Parquet</a>}{agentExports.experiment_manifest && <a href={AGENT_API + "/api/agent-runs/" + run + "/exports/experiment_manifest"}>Download experiment manifest</a>}</div>}</div>
      <div className="panel telemetry"><span>AGENT CONTEXT</span><p>{state?.goal ?? "Start a run to load the scenario goal."}</p><b>CURRENT ACTION</b><p>{lastAction}</p><b>INVENTORY</b><p>{state?.agent.inventory.join(", ") || "empty"} · weight {state?.metrics.carried_weight ?? 0}/{state?.metrics.inventory_weight_capacity ?? 0}</p><b>STATUS</b><p>{state?.agent.status_effects.join(", ") || "normal"}</p><div className="metrics"><b>SCORE {state?.metrics.normalized_score.toFixed(1) ?? "0.0"}</b><b>TIME {state?.metrics.simulated_time ?? 0}</b><b>VISITED {state?.metrics.unique_cells_visited ?? 0}</b><b>COVERAGE {((state?.metrics.exploration_coverage ?? 0) * 100).toFixed(1)}%</b><b>DIVERSITY {state?.metrics.action_diversity ?? 0}</b><b>RESOURCE {((state?.metrics.resource_efficiency ?? 0) * 100).toFixed(1)}%</b><b>INVALID {state?.metrics.invalid_actions ?? 0}</b><b>HAZARD {state?.metrics.hazard_damage_taken ?? 0}</b><b>WASTE {state?.metrics.unnecessary_actions ?? 0}</b><b>DISCOVERY D/I {state?.metrics.discovered_doors ?? 0}/{state?.metrics.discovered_items ?? 0}</b><b>DISCOVERY H/N {state?.metrics.discovered_hazards ?? 0}/{state?.metrics.discovered_npcs ?? 0}</b></div></div>
      {scenarioInfo && <div className="panel telemetry"><span>SCENARIO</span><h3>{scenarioInfo.name} · v{scenarioInfo.version}</h3><p>{scenarioInfo.goal}</p><p>{scenarioInfo.width}×{scenarioInfo.height} · {scenarioInfo.max_steps} step limit{scenarioInfo.time_limit ? " · " + scenarioInfo.time_limit + " time limit" : ""}</p></div>}
      {(researchMetadata.worldManifest || researchMetadata.rewardConfig) && <div className="panel telemetry"><span>RESEARCH METADATA / PRIVILEGED</span>{researchMetadata.worldManifest && <><h3>Generated world · seed {researchMetadata.worldManifest.seed}</h3><p>Hash {researchMetadata.worldManifest.world_hash} · generator v{researchMetadata.worldManifest.generator_version} · attempt {researchMetadata.worldManifest.generation_attempt}</p><p>{researchMetadata.worldManifest.dimensions.x}×{researchMetadata.worldManifest.dimensions.y} · {researchMetadata.worldManifest.validation.solvable ? "solvable" : "validation failed"}</p></>}{researchMetadata.rewardConfig && <p>Reward: step {researchMetadata.rewardConfig.baseline_per_step}, discovery {researchMetadata.rewardConfig.discovery_bonus}, success {researchMetadata.rewardConfig.terminal_success}, failure {researchMetadata.rewardConfig.terminal_failure}</p>}</div>}
      <div className="panel telemetry"><span>AGENT PERCEPTION / FILTERED</span><p>Mode: {agentObservation?.observation_mode ?? "not loaded"} · local cells: {agentObservation?.visible_cells.length ?? 0}</p><p>{agentObservation?.perception_note ?? "Only the cells and entities below were supplied to the agent."}</p><ol>{agentObservation?.visible_cells.filter((cell) => cell.entities.length > 0).slice(0, 8).map((cell) => <li key={cell.relative_position.x + ":" + cell.relative_position.y}>({cell.relative_position.x},{cell.relative_position.y}) {cell.entities.map((entity) => entity.name ?? entity.id).join(", ")}</li>)}</ol>{!agentObservation?.visible_cells.some((cell) => cell.entities.length > 0) && <p className="empty">No nearby entities were observed.</p>}</div>
      <div className="panel telemetry"><span>AGENT DECISION</span>{agentDecision ? <><p>Step {agentDecision.step} · {agentDecision.provider}{agentDecision.model ? " / " + agentDecision.model : ""}{agentDecision.latency_ms !== undefined ? " · " + agentDecision.latency_ms + " ms" : ""}</p><b>{agentDecision.action.type.toUpperCase()}{agentDecision.action.direction ? " " + agentDecision.action.direction.toUpperCase() : agentDecision.action.target_id ? " " + agentDecision.action.target_id : agentDecision.action.item_id ? " " + agentDecision.action.item_id : ""}</b><p>{agentDecision.decision_summary || "No decision summary supplied."}</p></> : <p className="empty">No provider decision recorded yet.</p>}</div>
      <div className="panel controls"><span>MANUAL BASELINE</span><p>{controlMode === "agent" && !paused ? "Pause the agent to take one manual control step." : controlMode === "replay" ? "Replay is read-only." : "Human control is active."}</p><div><button disabled={!manualControlsEnabled} onClick={() => act("move", "north")}>↑</button></div><div><button disabled={!manualControlsEnabled} onClick={() => act("move", "west")}>←</button><button disabled={!manualControlsEnabled} onClick={() => act("move", "south")}>↓</button><button disabled={!manualControlsEnabled} onClick={() => act("move", "east")}>→</button></div><button disabled={!manualControlsEnabled} onClick={() => act("inspect")}>Inspect</button><button disabled={!manualControlsEnabled} onClick={() => act("pickup")}>Pick up</button><button disabled={!manualControlsEnabled} onClick={() => act("rest")}>Rest</button><button disabled={!manualControlsEnabled} onClick={() => act("wait")}>Wait</button><button disabled={!manualControlsEnabled} onClick={() => act("talk", undefined, "caretaker")}>Talk to caretaker</button><button disabled={!manualControlsEnabled} onClick={() => act("open", undefined, "key_locker")}>Open locker</button><button disabled={!manualControlsEnabled} onClick={() => act("open", undefined, "service_door")}>Open service door</button><button disabled={!manualControlsEnabled} onClick={() => act("open", undefined, "exit_door")}>Open exit</button>{state?.agent.inventory[0] && <><button disabled={!manualControlsEnabled} onClick={() => act("use_item", undefined, undefined, state.agent.inventory[0])}>Use {state.agent.inventory[0]}</button><button disabled={!manualControlsEnabled} onClick={() => act("drop", undefined, undefined, state.agent.inventory[0])}>Drop {state.agent.inventory[0]}</button></>}</div>
      <div className="panel"><span>EVENTS</span><input aria-label="Filter events" placeholder="Filter events" value={eventFilter} onChange={(event) => setEventFilter(event.target.value)} /><select aria-label="Event type" value={eventType} onChange={(event) => setEventType(event.target.value)}><option value="all">All types</option>{eventTypes.map((type) => <option key={type} value={type}>{type}</option>)}</select><p>{visibleEvents.length} matching events</p><ol>{visibleEvents.map((event, index) => <li key={event.event_id ? "id:" + event.event_id : "legacy:" + index}><small>{event.type ?? "untyped"}{event.step !== undefined ? " · step " + event.step : ""}</small> <span>{event.message}</span></li>)}</ol>{!visibleEvents.length && <p>No matching events.</p>}</div>
      {frames.length > 1 && <div className="panel charts"><span>RESOURCE HISTORY</span>{(["health","energy","hydration"] as const).map((metric) => <div key={metric}><b>{metric.toUpperCase()}</b><div className={"spark " + metric}>{frames.slice(-24).map((frame,index) => <i key={index} title={metric + " " + frame.agent[metric]} style={{height: frame.agent[metric] + "%"}} />)}</div></div>)}</div>}
      {frames.length > 0 && <div className="panel playback"><span>REPLAY CONTROLS</span><p>Frame {frameIndex + 1} / {frames.length}</p><input aria-label="Replay frame" type="range" min="0" max={frames.length - 1} value={frameIndex} onChange={(event) => selectFrame(Number(event.target.value))} /><button onClick={() => selectFrame(0)}>|◀</button><button onClick={() => selectFrame(Math.max(0, frameIndex - 1))}>◀</button><button onClick={() => setPlaying((value) => !value)}>{playing ? "Pause" : "Play"}</button><button onClick={() => selectFrame(Math.min(frames.length - 1, frameIndex + 1))}>▶</button><button onClick={() => selectFrame(frames.length - 1)}>▶|</button><button onClick={() => setSpeed((value) => value === 4 ? 1 : value * 2)}>{speed}×</button></div>}
      <div className="panel"><span>REPLAY LIBRARY</span><input aria-label="Search replays" placeholder="Search run ID or outcome" value={replaySearch} onChange={(event) => setReplaySearch(event.target.value)} /><button onClick={() => void refreshRuns()}>Refresh runs</button>{visibleRuns.slice(0, 8).map((saved) => <button className="replay" key={saved.run_id} onClick={() => void loadReplay(saved.run_id)}>{saved.done ? saved.terminal_reason ?? "completed" : "active"} · step {saved.step} · {saved.run_id.slice(0,8)}</button>)}{!visibleRuns.length && <p className="empty">No matching runs.</p>}</div>
      {error && <p className="error">{error}</p>}</aside>
    </section>
  </main>;
}
