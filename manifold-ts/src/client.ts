/**
 * ManifoldClient — lightweight HTTP client for the MANIFOLD Trust API.
 *
 * All methods map 1-to-1 to the endpoints defined in the Phase 23 OpenAPI spec:
 *
 *   POST /shield          → evaluateShield()
 *   POST /b2b/handshake   → b2bHandshake()
 *   GET  /reputation/:id  → getReputation()
 *   POST /recruit         → recruit()
 *   GET  /policy          → getPolicy()
 *
 * Zero runtime dependencies — uses the built-in `fetch` API (Node ≥ 18).
 */

import type {
  AgentTrustScore,
  BrainTask,
  HandshakeResult,
  InterceptResult,
  ManifoldClientOptions,
  ManifoldError,
  OrgPolicy,
  RecruitmentRequest,
  RecruitmentResult,
  ReputationScore,
  ToolRegistration,
  TrustSignal,
} from "./types.js";

const DEFAULT_BASE_URL = "http://localhost:8080";
const DEFAULT_TIMEOUT_MS = 10_000;

/** Error thrown when the MANIFOLD API returns a non-2xx response. */
export class ManifoldApiError extends Error {
  readonly status: number;
  readonly body: ManifoldError;

  constructor(status: number, body: ManifoldError) {
    super(`MANIFOLD API error ${status}: ${body.message}`);
    this.name = "ManifoldApiError";
    this.status = status;
    this.body = body;
  }
}

/** Lightweight client for all MANIFOLD REST endpoints. */
export class ManifoldClient {
  private readonly baseUrl: string;
  private readonly headers: Record<string, string>;
  private readonly timeoutMs: number;

  constructor(options: ManifoldClientOptions = {}) {
    this.baseUrl = (options.baseUrl ?? DEFAULT_BASE_URL).replace(/\/$/, "");
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.headers = {
      "Content-Type": "application/json",
      Accept: "application/json",
      ...(options.apiKey ? { Authorization: `Bearer ${options.apiKey}` } : {}),
    };
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private async request<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
  ): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);

    let response: Response;
    try {
      response = await fetch(`${this.baseUrl}${path}`, {
        method,
        headers: this.headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
    }

    if (!response.ok) {
      let errorBody: ManifoldError;
      try {
        errorBody = (await response.json()) as ManifoldError;
      } catch {
        errorBody = { code: response.status, message: response.statusText };
      }
      throw new ManifoldApiError(response.status, errorBody);
    }

    return response.json() as Promise<T>;
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Evaluate a {@link BrainTask} through the `ActiveInterceptor` (`@shield`).
   *
   * Returns an {@link InterceptResult}.  If the result has `vetoed: true`, the
   * task **must not** be executed by the caller.
   *
   * @example
   * ```ts
   * const result = await client.evaluateShield({
   *   prompt: "Transfer $50,000 to account X",
   *   domain: "finance",
   *   stakes: 0.9,
   *   complexity: 0.6,
   * });
   * if (result.vetoed) {
   *   console.error("Task blocked:", result.reason);
   * }
   * ```
   */
  evaluateShield(task: BrainTask): Promise<InterceptResult> {
    return this.request<InterceptResult>("POST", "/shield", task);
  }

  /**
   * Perform a B2B policy handshake with a remote organisation.
   *
   * Only proceed with the cross-org API call if
   * {@link HandshakeResult.compatible} is `true`.
   *
   * @example
   * ```ts
   * const result = await client.b2bHandshake({
   *   org_id: "partner-corp",
   *   min_reliability: 0.85,
   *   max_risk: 0.2,
   *   domain: "finance",
   * });
   * if (!result.compatible) throw new Error(result.reason);
   * ```
   */
  b2bHandshake(remotePolicy: OrgPolicy): Promise<HandshakeResult> {
    return this.request<HandshakeResult>("POST", "/b2b/handshake", remotePolicy);
  }

  /**
   * Query the live reliability score for a tool or org from the
   * `ReputationHub`.
   *
   * @param agentId - Tool or organisation identifier (e.g. `"wolfram-alpha"`).
   *
   * @example
   * ```ts
   * const rep = await client.getReputation("wolfram-alpha");
   * console.log(`Reliability: ${rep.reliability}`);
   * ```
   */
  getReputation(agentId: string): Promise<ReputationScore> {
    const encoded = encodeURIComponent(agentId);
    return this.request<ReputationScore>("GET", `/reputation/${encoded}`);
  }

  /**
   * Ask the `SovereignRecruiter` to find the best-fit tool for a task.
   *
   * @example
   * ```ts
   * const result = await client.recruit({
   *   task_description: "Summarise a 100-page legal document",
   *   min_reliability: 0.9,
   *   required_capabilities: ["summarisation", "legal"],
   * });
   * if (result.recruited) console.log("Tool:", result.tool_id);
   * ```
   */
  recruit(request: RecruitmentRequest): Promise<RecruitmentResult> {
    return this.request<RecruitmentResult>("POST", "/recruit", request);
  }

  /**
   * Retrieve the local organisation's published `OrgPolicy`.
   *
   * @example
   * ```ts
   * const policy = await client.getPolicy();
   * console.log(`Domain: ${policy.domain}, max_risk: ${policy.max_risk}`);
   * ```
   */
  getPolicy(): Promise<OrgPolicy> {
    return this.request<OrgPolicy>("GET", "/policy");
  }

  // -------------------------------------------------------------------------
  // Agent Trust Score (ATS) — Phase 70
  // -------------------------------------------------------------------------

  /**
   * Retrieve the Agent Trust Score for a specific tool.
   *
   * @param toolId - The unique tool identifier (e.g. `"billing-api-v2"`).
   */
  getToolScore(toolId: string): Promise<AgentTrustScore> {
    return this.request<AgentTrustScore>(
      "GET",
      `/ats/score/${encodeURIComponent(toolId)}`,
    );
  }

  /**
   * Retrieve the ATS leaderboard (top 10 tools by score).
   */
  getLeaderboard(): Promise<AgentTrustScore[]> {
    return this.request<AgentTrustScore[]>("GET", "/ats/leaderboard");
  }

  /**
   * Register a tool in the ATS network. Requires authentication.
   *
   * @param tool - Tool registration details.
   */
  registerTool(tool: ToolRegistration): Promise<void> {
    return this.request<void>("POST", "/ats/register", tool);
  }

  /**
   * Submit a trust signal for a tool. Requires authentication.
   *
   * @param signal - The trust signal to record.
   */
  submitSignal(signal: TrustSignal): Promise<void> {
    return this.request<void>("POST", "/ats/signal", signal);
  }
}
