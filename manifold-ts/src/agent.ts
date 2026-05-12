/**
 * agent.ts — ManifoldAgentSDK
 *
 * TypeScript equivalent of manifold/sdk.py.
 *
 * Handles agent registration, periodic heartbeats, and bidirectional
 * governance-command polling.  Zero external dependencies — uses the
 * built-in `fetch` API (Node ≥ 18 / modern browsers).
 *
 * @example
 * ```ts
 * const sdk = new ManifoldAgentSDK(
 *   "finance-bot", "Finance Bot", ["billing", "refund"],
 *   "http://localhost:8080", "your-api-key",
 *   { domain: "finance" },
 * );
 * await sdk.register();
 * sdk.on("pause",   () => agent.pause());
 * sdk.on("resume",  () => agent.resume());
 * sdk.startHeartbeat();
 * sdk.startPolling();
 * // ...later
 * sdk.stop();
 * ```
 */

import type { AgentRecord, AgentCommand } from "./types.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Options accepted by {@link ManifoldAgentSDK}. */
export interface ManifoldAgentOptions {
  /**
   * Operational domain for the agent (e.g. `"finance"`, `"home"`).
   * @default "general"
   */
  domain?: string;
  /**
   * Heartbeat interval in milliseconds.
   * @default 30000
   */
  heartbeatIntervalMs?: number;
  /**
   * Timeout for regular HTTP requests (register, heartbeat) in milliseconds.
   * @default 10000
   */
  timeoutMs?: number;
  /**
   * Timeout for the long-poll command-fetch request in milliseconds.
   * @default 25000
   */
  pollingTimeoutMs?: number;
}

/**
 * Callback invoked when a governance command arrives for this agent.
 *
 * May be synchronous or asynchronous.
 */
export type CommandHandler = (
  payload: Record<string, unknown>,
) => void | Promise<void>;

// ---------------------------------------------------------------------------
// ManifoldAgentSDK
// ---------------------------------------------------------------------------

/**
 * Drop-in SDK for AI agents and physical robots to integrate with MANIFOLD.
 *
 * Mirrors `manifold/sdk.py` exactly.  All HTTP calls use `fetch` (Node ≥ 18).
 */
export class ManifoldAgentSDK {
  /** Unique agent identifier. */
  readonly agentId: string;
  /** Human-readable display name. */
  readonly displayName: string;
  /** List of capability tags used by TaskRouter for routing. */
  readonly capabilities: string[];
  /** Operational domain. */
  readonly domain: string;

  private readonly _baseUrl: string;
  private readonly _apiKey: string;
  private readonly _heartbeatIntervalMs: number;
  private readonly _timeoutMs: number;
  private readonly _pollingTimeoutMs: number;

  private readonly _handlers = new Map<string, CommandHandler>();
  private _running = false;
  private _heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private _pollingController: AbortController | null = null;

  constructor(
    agentId: string,
    displayName: string,
    capabilities: string[],
    baseUrl: string,
    apiKey: string,
    options: ManifoldAgentOptions = {},
  ) {
    this.agentId = agentId;
    this.displayName = displayName;
    this.capabilities = capabilities;
    this._baseUrl = baseUrl.replace(/\/$/, "");
    this._apiKey = apiKey;
    this.domain = options.domain ?? "general";
    this._heartbeatIntervalMs = options.heartbeatIntervalMs ?? 30_000;
    this._timeoutMs = options.timeoutMs ?? 10_000;
    this._pollingTimeoutMs = options.pollingTimeoutMs ?? 25_000;
  }

  // -------------------------------------------------------------------------
  // Private helpers
  // -------------------------------------------------------------------------

  private get _headers(): Record<string, string> {
    return {
      Authorization: `Bearer ${this._apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
    };
  }

  private async _fetch<T>(
    method: "GET" | "POST",
    path: string,
    body?: unknown,
    timeoutMs?: number,
    outerSignal?: AbortSignal,
  ): Promise<{ status: number; data: T }> {
    const controller = new AbortController();
    const timer = setTimeout(
      () => controller.abort(),
      timeoutMs ?? this._timeoutMs,
    );

    // Forward abort from an outer signal (e.g. the polling controller) so
    // that stop() immediately cancels in-flight long-poll fetches.
    let outerHandler: (() => void) | undefined;
    if (outerSignal) {
      outerHandler = () => controller.abort();
      outerSignal.addEventListener("abort", outerHandler, { once: true });
    }

    let response: Response;
    try {
      response = await fetch(`${this._baseUrl}${path}`, {
        method,
        headers: this._headers,
        body: body !== undefined ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
    } finally {
      clearTimeout(timer);
      if (outerSignal && outerHandler) {
        outerSignal.removeEventListener("abort", outerHandler);
      }
    }

    let data: T;
    try {
      data = (await response.json()) as T;
    } catch {
      data = {} as T;
    }
    return { status: response.status, data };
  }

  // -------------------------------------------------------------------------
  // Public API
  // -------------------------------------------------------------------------

  /**
   * Register this agent with the MANIFOLD server.
   *
   * @returns The server's {@link AgentRecord} for this agent.
   *
   * @example
   * ```ts
   * const record = await sdk.register();
   * console.log(record.status); // "active"
   * ```
   */
  async register(): Promise<AgentRecord> {
    const { data } = await this._fetch<AgentRecord>(
      "POST",
      "/agents/register",
      {
        agent_id: this.agentId,
        display_name: this.displayName,
        capabilities: this.capabilities,
        domain: this.domain,
      },
    );
    return data;
  }

  /**
   * Send a single heartbeat to the MANIFOLD server.
   *
   * @param status - Lifecycle status to report.
   * @returns `true` on HTTP 200, `false` on any error or non-200 response.
   */
  async heartbeat(status: AgentRecord["status"] = "active"): Promise<boolean> {
    try {
      const { status: httpStatus } = await this._fetch<unknown>(
        "POST",
        `/agents/${encodeURIComponent(this.agentId)}/heartbeat`,
        { status },
        5_000,
      );
      return httpStatus === 200;
    } catch {
      return false;
    }
  }

  /**
   * Register a callback for a governance command name.
   *
   * Calling `on()` a second time with the same command replaces the previous
   * handler.
   *
   * @param command - Command string (e.g. `"pause"`, `"yield"`, `"redirect"`).
   * @param handler - Called with the command payload when the command arrives.
   *
   * @example
   * ```ts
   * sdk.on("pause",    ()  => robot.stop());
   * sdk.on("redirect", (p) => robot.moveTo(p.target as string));
   * ```
   */
  on(command: string, handler: CommandHandler): void {
    this._handlers.set(command, handler);
  }

  /**
   * Start the periodic heartbeat background loop.
   *
   * Safe to call multiple times — subsequent calls are no-ops.
   * The interval timer is unreffed in Node.js so it does not keep the process
   * alive when nothing else is running.
   *
   * @param intervalMs - Override the heartbeat interval (defaults to the
   *   value set in the constructor options).
   */
  startHeartbeat(intervalMs?: number): void {
    if (this._heartbeatTimer !== null) return; // already running
    this._running = true;
    const interval = intervalMs ?? this._heartbeatIntervalMs;
    this._heartbeatTimer = setInterval(() => {
      void this.heartbeat().catch(() => undefined);
    }, interval);
    // Prevent the timer from keeping the Node.js process alive alone.
    const timer = this._heartbeatTimer as NodeJS.Timeout | undefined;
    if (timer && typeof timer === "object" && "unref" in timer) {
      (timer as NodeJS.Timeout).unref();
    }
  }

  /**
   * Start the command polling background loop.
   *
   * Polls `GET /agents/{id}/commands` continuously, dispatching received
   * commands to handlers registered with {@link on}.  Backs off 5 s after
   * transient errors.  Safe to call multiple times — subsequent calls are
   * no-ops.
   */
  startPolling(): void {
    if (this._pollingController !== null) return; // already running
    this._running = true;
    this._pollingController = new AbortController();
    void this._pollLoop(this._pollingController.signal);
  }

  private async _pollLoop(signal: AbortSignal): Promise<void> {
    while (this._running && !signal.aborted) {
      try {
        const { status, data } = await this._fetch<{
          commands: AgentCommand[];
          agent_id: string;
        }>(
          "GET",
          `/agents/${encodeURIComponent(this.agentId)}/commands`,
          undefined,
          this._pollingTimeoutMs,
          signal,
        );

        if (status === 200) {
          for (const cmd of data.commands ?? []) {
            const handler = this._handlers.get(cmd.command);
            if (handler) {
              try {
                await handler(cmd.payload ?? {});
              } catch (err) {
                console.warn(
                  `[ManifoldAgentSDK] Handler error for "${cmd.command}":`,
                  err,
                );
              }
            }
          }
        }
      } catch {
        if (signal.aborted) break;
        // Transient error — back off before retrying.
        await new Promise<void>((resolve) => setTimeout(resolve, 5_000));
      }
    }
  }

  /**
   * Stop the heartbeat timer and the polling loop.
   *
   * Safe to call multiple times.
   */
  stop(): void {
    this._running = false;
    if (this._heartbeatTimer !== null) {
      clearInterval(this._heartbeatTimer);
      this._heartbeatTimer = null;
    }
    if (this._pollingController !== null) {
      this._pollingController.abort();
      this._pollingController = null;
    }
  }
}
