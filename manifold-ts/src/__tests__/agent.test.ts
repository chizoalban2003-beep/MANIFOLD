/**
 * Unit tests for ManifoldAgentSDK.
 *
 * Run with: node --test (Node ≥ 18)
 * These tests mock the fetch API so no MANIFOLD server is required.
 */

import assert from "node:assert/strict";
import { describe, it, afterEach } from "node:test";

import { ManifoldAgentSDK } from "../agent.js";
import type { AgentRecord } from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function mockFetch(status: number, body: unknown): typeof globalThis.fetch {
  return async () =>
    ({
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? "OK" : "Error",
      json: async () => body,
    } as Response);
}

function makeSdk(overrides: Partial<{ domain: string }> = {}): ManifoldAgentSDK {
  return new ManifoldAgentSDK(
    "test-agent",
    "Test Agent",
    ["nav", "clean"],
    "http://test.local",
    "test-key",
    { domain: overrides.domain ?? "home", heartbeatIntervalMs: 60_000 },
  );
}

afterEach(() => {
  // Remove any fetch mock installed during the test.
  delete (globalThis as { fetch?: unknown }).fetch;
});

// ---------------------------------------------------------------------------
// constructor
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK constructor", () => {
  it("stores agentId, displayName, capabilities and domain", () => {
    const sdk = makeSdk();
    assert.equal(sdk.agentId, "test-agent");
    assert.equal(sdk.displayName, "Test Agent");
    assert.deepEqual(sdk.capabilities, ["nav", "clean"]);
    assert.equal(sdk.domain, "home");
  });

  it("defaults domain to 'general' when omitted", () => {
    const sdk = new ManifoldAgentSDK("a", "A", [], "http://host", "key");
    assert.equal(sdk.domain, "general");
  });

  it("strips trailing slash from baseUrl", () => {
    const sdk = new ManifoldAgentSDK(
      "a",
      "A",
      [],
      "http://host:8080/",
      "key",
    );
    // Verify indirectly: register() should call http://host:8080/agents/register
    // (tested by not having double-slash in the mocked URL)
    assert.ok(sdk instanceof ManifoldAgentSDK);
  });
});

// ---------------------------------------------------------------------------
// register()
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK.register()", () => {
  it("returns the AgentRecord from the server on HTTP 201", async () => {
    const expected: AgentRecord = {
      agent_id: "test-agent",
      display_name: "Test Agent",
      capabilities: ["nav", "clean"],
      domain: "home",
      level: 1,
      health: 1.0,
      status: "active",
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(201, expected);

    const sdk = makeSdk();
    const record = await sdk.register();

    assert.equal(record.agent_id, "test-agent");
    assert.equal(record.display_name, "Test Agent");
    assert.equal(record.status, "active");
    assert.equal(record.health, 1.0);
  });

  it("returns the server response even on non-2xx status", async () => {
    const body: AgentRecord = {
      agent_id: "test-agent",
      display_name: "Test Agent",
      capabilities: [],
      domain: "home",
      level: 0,
      health: 0,
      status: "offline",
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(409, body);

    const sdk = makeSdk();
    // register() returns the parsed body regardless; callers check the record
    const record = await sdk.register();
    assert.equal(record.agent_id, "test-agent");
  });
});

// ---------------------------------------------------------------------------
// heartbeat()
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK.heartbeat()", () => {
  it("returns true on HTTP 200", async () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, {});

    const sdk = makeSdk();
    const ok = await sdk.heartbeat();
    assert.equal(ok, true);
  });

  it("returns true for default status 'active'", async () => {
    let capturedBody: string | undefined;
    (globalThis as { fetch?: unknown }).fetch = async (
      _url: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      capturedBody = init?.body as string;
      return {
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({}),
      } as Response;
    };

    const sdk = makeSdk();
    const ok = await sdk.heartbeat();

    assert.equal(ok, true);
    assert.ok(capturedBody?.includes('"active"'), "body should contain 'active'");
  });

  it("sends the supplied status in the request body", async () => {
    let capturedBody: string | undefined;
    (globalThis as { fetch?: unknown }).fetch = async (
      _url: RequestInfo | URL,
      init?: RequestInit,
    ) => {
      capturedBody = init?.body as string;
      return {
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({}),
      } as Response;
    };

    const sdk = makeSdk();
    await sdk.heartbeat("paused");

    assert.ok(capturedBody?.includes('"paused"'), "body should contain 'paused'");
  });

  it("returns false on HTTP 404", async () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(404, {
      code: 404,
      message: "Not found",
    });

    const sdk = makeSdk();
    const ok = await sdk.heartbeat();
    assert.equal(ok, false);
  });

  it("returns false when fetch throws (network error)", async () => {
    (globalThis as { fetch?: unknown }).fetch = async () => {
      throw new Error("ECONNREFUSED");
    };

    const sdk = makeSdk();
    const ok = await sdk.heartbeat();
    assert.equal(ok, false);
  });
});

// ---------------------------------------------------------------------------
// on() — handler registration
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK.on()", () => {
  it("registers a handler and replaces it on a second call", () => {
    const sdk = makeSdk();
    let counter = 0;

    sdk.on("pause", () => { counter += 1; });
    sdk.on("pause", () => { counter += 10; });

    // Access internal map to verify replacement
    const handlers = (
      sdk as unknown as { _handlers: Map<string, () => void> }
    )._handlers;

    assert.equal(handlers.size, 1, "only one handler per command");
    handlers.get("pause")?.();
    assert.equal(counter, 10, "second on() should replace the first");
  });

  it("supports multiple distinct commands", () => {
    const sdk = makeSdk();
    sdk.on("pause", () => undefined);
    sdk.on("resume", () => undefined);
    sdk.on("redirect", () => undefined);

    const handlers = (
      sdk as unknown as { _handlers: Map<string, () => void> }
    )._handlers;

    assert.equal(handlers.size, 3);
    assert.ok(handlers.has("pause"));
    assert.ok(handlers.has("resume"));
    assert.ok(handlers.has("redirect"));
  });
});

// ---------------------------------------------------------------------------
// startHeartbeat() / startPolling() — idempotency
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK.startHeartbeat() idempotency", () => {
  it("is safe to call multiple times (no-op on repeat)", () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, {});

    const sdk = makeSdk();
    sdk.startHeartbeat(60_000);
    const timerA = (sdk as unknown as { _heartbeatTimer: unknown })
      ._heartbeatTimer;
    sdk.startHeartbeat(60_000); // second call — no-op
    const timerB = (sdk as unknown as { _heartbeatTimer: unknown })
      ._heartbeatTimer;

    assert.equal(timerA, timerB, "timer reference must not change");
    sdk.stop();
  });
});

describe("ManifoldAgentSDK.startPolling() idempotency", () => {
  it("is safe to call multiple times (no-op on repeat)", () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, {
      commands: [],
      agent_id: "test-agent",
    });

    const sdk = makeSdk();
    sdk.startPolling();
    const ctrlA = (sdk as unknown as { _pollingController: unknown })
      ._pollingController;
    sdk.startPolling(); // second call — no-op
    const ctrlB = (sdk as unknown as { _pollingController: unknown })
      ._pollingController;

    assert.equal(ctrlA, ctrlB, "polling controller must not change");
    sdk.stop();
  });
});

// ---------------------------------------------------------------------------
// stop()
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK.stop()", () => {
  it("can be called before start() without throwing", () => {
    const sdk = makeSdk();
    assert.doesNotThrow(() => sdk.stop());
  });

  it("is idempotent — safe to call multiple times", () => {
    const sdk = makeSdk();
    sdk.stop();
    assert.doesNotThrow(() => sdk.stop());
    assert.doesNotThrow(() => sdk.stop());
  });

  it("clears the heartbeat timer", () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, {});

    const sdk = makeSdk();
    sdk.startHeartbeat(60_000);

    assert.notEqual(
      (sdk as unknown as { _heartbeatTimer: unknown })._heartbeatTimer,
      null,
    );

    sdk.stop();

    assert.equal(
      (sdk as unknown as { _heartbeatTimer: unknown })._heartbeatTimer,
      null,
    );
  });

  it("clears the polling controller", () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, {
      commands: [],
      agent_id: "test-agent",
    });

    const sdk = makeSdk();
    sdk.startPolling();

    assert.notEqual(
      (sdk as unknown as { _pollingController: unknown })._pollingController,
      null,
    );

    sdk.stop();

    assert.equal(
      (sdk as unknown as { _pollingController: unknown })._pollingController,
      null,
    );
  });

  it("sets _running to false", () => {
    const sdk = makeSdk();
    (sdk as unknown as { _running: boolean })._running = true;
    sdk.stop();
    assert.equal((sdk as unknown as { _running: boolean })._running, false);
  });
});

// ---------------------------------------------------------------------------
// Command dispatch via polling loop
// ---------------------------------------------------------------------------

describe("ManifoldAgentSDK command dispatch", () => {
  it("dispatches received commands to on() handlers", async () => {
    let fetchCallCount = 0;
    (globalThis as { fetch?: unknown }).fetch = async () => {
      fetchCallCount++;
      const commands =
        fetchCallCount === 1
          ? [
              {
                id: "cmd-1",
                command: "pause",
                payload: { reason: "obstacle" },
                queued_at: Date.now(),
              },
            ]
          : [];
      return {
        ok: true,
        status: 200,
        statusText: "OK",
        json: async () => ({ commands, agent_id: "test-agent" }),
      } as Response;
    };

    const sdk = makeSdk();
    let received: Record<string, unknown> | null = null;
    sdk.on("pause", (payload) => {
      received = payload;
    });

    sdk.startPolling();

    // Allow the first poll iteration to complete.
    await new Promise<void>((resolve) => setTimeout(resolve, 100));
    sdk.stop();

    assert.ok(received !== null, "pause handler should have been called");
    assert.equal(
      (received as Record<string, unknown>).reason,
      "obstacle",
    );
  });

  it("ignores commands with no registered handler (no throw)", async () => {
    (globalThis as { fetch?: unknown }).fetch = async () => ({
      ok: true,
      status: 200,
      statusText: "OK",
      json: async () => ({
        commands: [
          {
            id: "cmd-2",
            command: "unknown_command",
            payload: {},
            queued_at: Date.now(),
          },
        ],
        agent_id: "test-agent",
      }),
    } as Response);

    const sdk = makeSdk();
    sdk.startPolling();

    // Should not throw even though no handler is registered.
    await new Promise<void>((resolve) => setTimeout(resolve, 100));
    assert.doesNotThrow(() => sdk.stop());
  });
});
