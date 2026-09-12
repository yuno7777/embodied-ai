import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import BenchmarksPage from "./page";

const summary = { total_runs: 2, completed_runs: 2, escaped_runs: 1, success_rate: .5, mean_steps: 6, mean_simulated_time: 7, mean_normalized_score: 60, mean_final_health: 90, mean_invalid_actions: .5, mean_hazard_damage: 2, total_provider_calls: 12, mean_provider_latency_ms: 4, total_input_tokens: 100, total_output_tokens: 20 };
const first = { run_id: "aaaaaaaa-1", step: 5, done: true, terminal_reason: "escaped", agent: { health: 100 }, metrics: { normalized_score: 80, invalid_actions: 0, hazard_damage_taken: 0 } };
const second = { run_id: "bbbbbbbb-2", step: 7, done: true, terminal_reason: "timeout", agent: { health: 80 }, metrics: { normalized_score: 40, invalid_actions: 1, hazard_damage_taken: 4 } };
const response = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

it("compares two authoritative completed runs with directional deltas", async () => {
  vi.stubGlobal("fetch", vi.fn((url: string) => response(url.endsWith("/api/benchmarks") ? summary : url.endsWith("/api/replays") ? [] : [first, second])));
  render(<BenchmarksPage />);
  await waitFor(() => expect(screen.getByText("SCORE Δ -40.0")).toBeInTheDocument());
  expect(screen.getByText("STEPS Δ 2.0")).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Right run"), { target: { value: first.run_id } });
  expect(screen.getByText("SCORE Δ 0.0")).toBeInTheDocument();
});
