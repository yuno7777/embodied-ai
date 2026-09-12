import "@testing-library/jest-dom/vitest";

export class MockWebSocket {
  static instances: MockWebSocket[] = [];
  onopen: (() => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  constructor(_url: string) { MockWebSocket.instances.push(this); }
  close() { this.onclose?.(); }
  open() { this.onopen?.(); }
  message(data: unknown) { this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent); }
}

Object.defineProperty(window, "WebSocket", { value: MockWebSocket, writable: true });
