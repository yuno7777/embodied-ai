import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import Observer from "./observer";
import { MockWebSocket } from "./test-setup";

const snapshot = {
  run_id: "run-1", step: 0, goal: "Escape safely", width: 3, height: 3, walls: [],
  agent: { position: { x: 1, y: 1 }, facing: "east", health: 100, energy: 100, hydration: 100, inventory: [], status_effects: [], alive: true, escaped: false },
  items: [], hazards: [], doors: [], containers: [], done: false,
  metrics: { normalized_score: 0, invalid_actions: 0, unique_cells_visited: 1, hazard_damage_taken: 0, simulated_time: 0, unnecessary_actions: 0, recovery_after_failure: false },
};
const observation = {
  step: 0, observation_mode: "minimal", recent_events: [],
  visible_cells: [{ relative_position: { x: 1, y: 0 }, terrain: "floor", entities: [{ id: "visible_door", type: "door", state: "closed" }] }],
};

function response(body: unknown) { return Promise.resolve(new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } })); }

describe("Observer", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); MockWebSocket.instances = []; });

  it("renders the researcher and manual-control views", () => {
    vi.stubGlobal("fetch", vi.fn((url: string) => response(url.endsWith("/api/replays") ? [] : [])));
    render(<Observer />);
    expect(screen.getByText("RESEARCHER VIEW")).toBeInTheDocument();
    expect(screen.getByText("MANUAL BASELINE")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Inspect" })).toBeInTheDocument();
    expect(screen.getByText("AGENT PERCEPTION / FILTERED")).toBeInTheDocument();
  });

  it("creates a manual run and exposes its embodied state", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/runs") && init?.method === "POST") return response({ run_id: "run-1", snapshot, observation, world_manifest: { seed: 99, world_hash: "fnv1a64:test", generator_version: 1, generation_attempt: 0, dimensions: { x: 11, y: 7 }, validation: { solvable: true } }, reward_config: { baseline_per_step: -1, discovery_bonus: 5, invalid_action_penalty: -2, terminal_success: 100, terminal_failure: -100 } });
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.change(screen.getByLabelText("Observation mode"), { target: { value: "rich" } });
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(screen.getByText("active")).toBeInTheDocument());
    expect(screen.getByText("FACING east")).toBeInTheDocument();
    const createRequest = fetchMock.mock.calls.find(([url, init]) => url.endsWith("/api/runs") && init?.method === "POST");
    expect(JSON.parse(String(createRequest?.[1]?.body)).controller).toBe("manual");
    expect(screen.getByText("Escape safely")).toBeInTheDocument();
    expect(screen.getByText("Generated world · seed 99")).toBeInTheDocument();
    expect(screen.getByText(/Hash fnv1a64:test/)).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("visible_door");
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith("/api/runs") && JSON.parse(String((init as RequestInit).body)).observation_mode === "rich")).toBe(true);
  });

  it("uses the paused-only manual-step endpoint for human control", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/runs") && init?.method === "POST") return response({ run_id: "run-1", snapshot });
      if (url.endsWith("/manual-step")) return response({ events: [] });
      if (url.endsWith("/pause")) return response({ paused: true });
      if (url.endsWith("/api/runs/run-1")) return response(snapshot);
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(screen.getByText("active")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Pause" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Resume" })).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "Wait" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/manual-step"))).toBe(true));
  });

  it("renders typed WebSocket updates from the authoritative server", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/runs") && init?.method === "POST") return response({ run_id: "run-1", snapshot, observation });
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
    MockWebSocket.instances[0].open();
    MockWebSocket.instances[0].message({ type: "agent_decision", decision: { step: 1, action: { type: "wait" }, decision_summary: "Pausing to assess the room.", provider: "scripted", latency_ms: 0 }, result: { events: [{ message: "The body waited." }], observation } });
    await waitFor(() => expect(screen.getByText("LIVE LINK: connected")).toBeInTheDocument());
    expect(screen.getByText("The body waited.")).toBeInTheDocument();
    expect(screen.getByText("Pausing to assess the room.")).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")[0]).toHaveTextContent("visible_door");
  });

  it("refreshes authoritative state and reconnects after a live socket closes", async () => {
    const recoveredSnapshot = { ...snapshot, step: 2 };
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/runs") && init?.method === "POST") return response({ run_id: "run-1", snapshot, observation });
      if (url.endsWith("/api/runs/run-1/replay")) return response({ snapshot: recoveredSnapshot, observations: [observation] });
      if (url.endsWith("/api/runs/run-1")) return response(recoveredSnapshot);
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
    MockWebSocket.instances[0].close();
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(2));
    expect(screen.getByText("STEP 2")).toBeInTheDocument();
    expect(screen.getByText("LIVE LINK: reconnecting")).toBeInTheDocument();
  });

  it("synchronizes persisted decisions with replay frame navigation", async () => {
    const finalSnapshot = { ...snapshot, step: 1 };
    const decision = { step: 1, action: { type: "wait" }, decision_summary: "Check the room before moving.", provider: "scripted", latency_ms: 0 };
    const fetchMock = vi.fn((url: string) => {
      if (url.endsWith("/api/runs/run-1/replay")) return response({ snapshot: finalSnapshot, timeline: [snapshot, finalSnapshot], observations: [observation, observation], decisions: [decision], events: [] });
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer initialRunId="run-1" />);
    await waitFor(() => expect(screen.getByText("Check the room before moving.")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: "|◀" }));
    expect(screen.getByText("No provider decision recorded yet.")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "▶" }));
    expect(screen.getByText("Check the room before moving.")).toBeInTheDocument();
  });

  it("starts a selected provider through the separate local agent service", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (String(url).endsWith(":8090/api/agent-runs") && init?.method === "POST") return response({ run_id: "agent-1", status: "running" });
      if (String(url).endsWith(":8090/api/agent-runs/agent-1")) return response({ run_id: "agent-1", status: "completed", terminal_reason: "escaped", exports: { experiment_manifest: "data/runs/agent-1.experiment.json" } });
      return response(url.endsWith("/api/replays") ? [] : []);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.change(screen.getByLabelText("Provider"), { target: { value: "mock_reasoning" } });
    fireEvent.change(screen.getByLabelText("Seed"), { target: { value: "99" } });
    fireEvent.change(screen.getByLabelText("Memory window"), { target: { value: "12" } });
    fireEvent.change(screen.getByLabelText("Max steps"), { target: { value: "25" } });
    fireEvent.change(screen.getByLabelText("Memory mode"), { target: { value: "none" } });
    fireEvent.change(screen.getByLabelText("Model"), { target: { value: "test-model" } });
    fireEvent.click(screen.getByRole("button", { name: "Start agent" }));
    await waitFor(() => expect(screen.getByText("Waiting for mock_reasoning to choose the first simulated action.")).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText("LIVE LINK: reconnecting · AGENT: completed")).toBeInTheDocument());
    expect(screen.getByText("Pause the agent to take one manual control step.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Wait" })).toBeDisabled();
    expect(screen.getByRole("link", { name: "Download experiment manifest" })).toHaveAttribute("href", expect.stringContaining("/exports/experiment_manifest"));
    const request = fetchMock.mock.calls.find(([url]) => String(url).endsWith(":8090/api/agent-runs"));
    expect(request).toBeTruthy();
    expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ provider: "mock_reasoning", seed: 99, max_steps: 25, model: "test-model", observation_mode: "normal", memory_mode: "none", memory_window: 12 });
  });

  it("filters replay events by type and text, including older and untyped events", async () => {
    const events = [
      { event_id: "old", type: "hazard", step: 1, message: "Toxic gas damaged the body." },
      ...Array.from({ length: 15 }, (_, i) => ({ event_id: String(i), type: "movement", step: i + 2, message: "Moved east." })),
      { message: "Legacy replay entry." },
    ];
    vi.stubGlobal("fetch", vi.fn((url: string) => response(url.endsWith("/run-1/replay") ? { snapshot, events } : [])));
    render(<Observer initialRunId="run-1" />);
    await waitFor(() => expect(screen.getByText("Toxic gas damaged the body.")).toBeInTheDocument());
    fireEvent.change(screen.getByLabelText("Event type"), { target: { value: "hazard" } });
    expect(screen.queryByText("Moved east.")).not.toBeInTheDocument();
    expect(screen.getByText("1 matching events")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Filter events"), { target: { value: "TOXIC" } });
    expect(screen.getByText("Toxic gas damaged the body.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Filter events"), { target: { value: "water" } });
    expect(screen.getByText("No matching events.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Filter events"), { target: { value: "" } });
    fireEvent.change(screen.getByLabelText("Event type"), { target: { value: "untyped" } });
    expect(screen.getByText("Legacy replay entry.")).toBeInTheDocument();
  });

  it("preserves live event types and deduplicates repeated event IDs", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string, init?: RequestInit) => response(
      url.endsWith("/api/runs") && init?.method === "POST" ? { run_id: "run-1", snapshot } : []
    )));
    render(<Observer />);
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(MockWebSocket.instances).toHaveLength(1));
    const update = { type: "simulation_event", snapshot: { ...snapshot, step: 1, agent: { ...snapshot.agent, health: 90 } }, result: { events: [{ event_id: "e1", type: "hazard", message: "Fire damage." }] } };
    MockWebSocket.instances[0].message(update);
    MockWebSocket.instances[0].message(update);
    await waitFor(() => expect(screen.getAllByText("Fire damage.")).toHaveLength(1));
    expect(screen.getByText("STEP 1")).toBeInTheDocument();
    expect(screen.getByText("HEALTH 90")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Event type"), { target: { value: "observer" } });
    expect(screen.queryByText("Fire damage.")).not.toBeInTheDocument();
    expect(screen.getByText("Manual run started.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Event type"), { target: { value: "hazard" } });
    expect(screen.getByText("Fire damage.")).toBeInTheDocument();
  });

  it("keeps loaded active replays read-only and detached from live updates", async () => {
    vi.stubGlobal("fetch", vi.fn((url: string) => response(url.endsWith("/run-1/replay") ? { snapshot, events: [] } : [])));
    render(<Observer initialRunId="run-1" />);
    await waitFor(() => expect(screen.getByText("Loaded persisted replay.")).toBeInTheDocument());
    expect(screen.getByRole("button", { name: "Pause" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Abort" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Wait" })).toBeDisabled();
    expect(MockWebSocket.instances).toHaveLength(0);
  });

  it("supports keyboard movement only while manual control is enabled", async () => {
    const fetchMock = vi.fn((url: string, init?: RequestInit) => {
      if (url.endsWith("/api/runs") && init?.method === "POST") return response({ run_id: "run-1", snapshot, observation });
      if (url.endsWith("/step")) return response({ events: [], observation });
      if (url.endsWith("/api/runs/run-1")) return response(snapshot);
      return response(url.endsWith("/api/replays") ? [] : [snapshot]);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Observer />);
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/step"))).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Start manual" }));
    await waitFor(() => expect(screen.getByText("active")).toBeInTheDocument());
    fireEvent.keyDown(window, { key: "ArrowRight" });
    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/step"));
      expect(JSON.parse(String((request?.[1] as RequestInit).body))).toMatchObject({ type: "move", direction: "east" });
    });
  });
});
