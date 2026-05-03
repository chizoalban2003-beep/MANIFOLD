/**
 * Unit tests for ManifoldClient and shield utilities.
 *
 * Run with: node --test (Node ≥ 18)
 * These tests mock the fetch API so no MANIFOLD server is required.
 */

import assert from "node:assert/strict";
import { describe, it, beforeEach } from "node:test";

import { ManifoldClient, ManifoldApiError } from "../client.js";
import { createShield, ShieldVetoError } from "../shield.js";
import type { InterceptResult, OrgPolicy, ReputationScore } from "../types.js";

// ---------------------------------------------------------------------------
// Fetch mock helpers
// ---------------------------------------------------------------------------

function mockFetch(
  status: number,
  body: unknown,
): typeof globalThis.fetch {
  return async () =>
    ({
      ok: status >= 200 && status < 300,
      status,
      statusText: status === 200 ? "OK" : "Error",
      json: async () => body,
    } as Response);
}

// ---------------------------------------------------------------------------
// ManifoldClient
// ---------------------------------------------------------------------------

describe("ManifoldClient", () => {
  let client: ManifoldClient;

  beforeEach(() => {
    client = new ManifoldClient({ baseUrl: "http://test.local" });
  });

  it("evaluateShield returns InterceptResult on 200", async () => {
    const expected: InterceptResult = {
      vetoed: false,
      reason: "Risk within threshold",
      risk_score: 0.15,
      confidence: 0.92,
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, expected);

    const result = await client.evaluateShield({
      prompt: "Calculate 2+2",
      domain: "math",
      stakes: 0.1,
    });
    assert.equal(result.vetoed, false);
    assert.equal(result.risk_score, 0.15);
  });

  it("evaluateShield throws ManifoldApiError on 422", async () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(422, {
      code: 422,
      message: "Task vetoed",
    });

    await assert.rejects(
      () =>
        client.evaluateShield({ prompt: "DELETE all users", domain: "db", stakes: 1.0 }),
      ManifoldApiError,
    );
  });

  it("b2bHandshake returns HandshakeResult on 200", async () => {
    const expected = {
      compatible: true,
      reason: "Policies compatible",
      trust_score: 0.88,
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, expected);

    const policy: OrgPolicy = { org_id: "partner", min_reliability: 0.8, max_risk: 0.3 };
    const result = await client.b2bHandshake(policy);
    assert.equal(result.compatible, true);
    assert.equal(result.trust_score, 0.88);
  });

  it("b2bHandshake throws ManifoldApiError on 403", async () => {
    (globalThis as { fetch?: unknown }).fetch = mockFetch(403, {
      code: 403,
      message: "Policies incompatible",
    });

    await assert.rejects(
      () => client.b2bHandshake({ org_id: "bad-actor" }),
      ManifoldApiError,
    );
  });

  it("getReputation returns ReputationScore on 200", async () => {
    const expected: ReputationScore = {
      agent_id: "wolfram-alpha",
      reliability: 0.97,
      sample_count: 120,
      last_updated: "2026-05-03T06:00:00Z",
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, expected);

    const rep = await client.getReputation("wolfram-alpha");
    assert.equal(rep.agent_id, "wolfram-alpha");
    assert.equal(rep.reliability, 0.97);
  });

  it("getPolicy returns OrgPolicy on 200", async () => {
    const expected: OrgPolicy = {
      org_id: "my-org",
      min_reliability: 0.85,
      max_risk: 0.2,
      domain: "finance",
      version: "1.0.0",
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, expected);

    const policy = await client.getPolicy();
    assert.equal(policy.org_id, "my-org");
    assert.equal(policy.domain, "finance");
  });
});

// ---------------------------------------------------------------------------
// shield
// ---------------------------------------------------------------------------

describe("createShield", () => {
  it("passes through when not vetoed", async () => {
    const passResult: InterceptResult = {
      vetoed: false,
      reason: "OK",
      risk_score: 0.1,
      confidence: 0.95,
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, passResult);

    const client = new ManifoldClient({ baseUrl: "http://test.local" });
    const shield = createShield(client);

    let called = false;
    const safeFn = shield(
      async () => {
        called = true;
        return "done";
      },
      () => ({ prompt: "safe op", stakes: 0.1 }),
    );

    const result = await safeFn();
    assert.equal(called, true);
    assert.equal(result, "done");
  });

  it("throws ShieldVetoError when vetoed", async () => {
    const vetoResult: InterceptResult = {
      vetoed: true,
      reason: "Too risky",
      risk_score: 0.95,
      confidence: 0.99,
    };
    (globalThis as { fetch?: unknown }).fetch = mockFetch(200, vetoResult);

    const client = new ManifoldClient({ baseUrl: "http://test.local" });
    const shield = createShield(client);

    const riskyFn = shield(
      async () => "executed",
      () => ({ prompt: "risky op", stakes: 0.99 }),
    );

    await assert.rejects(() => riskyFn(), ShieldVetoError);
  });

  it("throws when server unreachable and failOpen=false (default)", async () => {
    (globalThis as { fetch?: unknown }).fetch = async () => {
      throw new Error("ECONNREFUSED");
    };

    const client = new ManifoldClient({ baseUrl: "http://unreachable.local" });
    const shield = createShield(client);
    const fn = shield(
      async () => "ok",
      () => ({ prompt: "test" }),
    );

    await assert.rejects(() => fn(), { name: "Error" });
  });

  it("proceeds when server unreachable and failOpen=true", async () => {
    (globalThis as { fetch?: unknown }).fetch = async () => {
      throw new Error("ECONNREFUSED");
    };

    const client = new ManifoldClient({ baseUrl: "http://unreachable.local" });
    const shield = createShield(client, { failOpen: true });
    const fn = shield(
      async () => "ok",
      () => ({ prompt: "test" }),
    );

    const result = await fn();
    assert.equal(result, "ok");
  });
});
