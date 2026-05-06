/**
 * shadow.ts — ShadowModeWrapper equivalent for Node.js agents.
 *
 * Provides the same non-destructive observation semantics as the Python
 * `ShadowModeWrapper` in `manifold/connector.py`.  When `active` is `false`,
 * the wrapper **never** modifies agent behaviour — it only records what
 * MANIFOLD *would* have decided.
 *
 * Usage
 * -----
 * ```ts
 * import { ShadowModeWrapper } from "@manifold/client";
 *
 * const shadow = new ShadowModeWrapper(client, { active: false });
 *
 * // Your existing agent function — unchanged behaviour
 * const result = await shadow.observe(
 *   async () => myAgent.run(prompt),
 *   { prompt, domain: "finance", stakes: 0.8 },
 * );
 *
 * // Retrieve the accumulated audit report
 * const report = shadow.getReport();
 * console.log(`Disagreements: ${report.disagreement_rate}`);
 * ```
 */

import type { BrainTask, ShadowReport, AdversarialSuspect, PenaltyProposal } from "./types.js";
import { ManifoldClient } from "./client.js";

// ---------------------------------------------------------------------------
// ShadowModeOptions
// ---------------------------------------------------------------------------

/** Configuration for {@link ShadowModeWrapper}. */
export interface ShadowModeOptions {
  /**
   * When `false` (default), the wrapper **only observes** — the wrapped
   * function always executes, and MANIFOLD decisions are recorded but not
   * enforced.
   *
   * When `true`, MANIFOLD decisions **are** enforced: a high-risk task will
   * throw a `ShadowVetoError` before the wrapped function runs.
   *
   * @default false
   */
  active?: boolean;

  /**
   * If `true`, failures to contact the MANIFOLD API are silently ignored and
   * the wrapped function runs regardless.
   *
   * @default true
   */
  failOpen?: boolean;
}

/** Thrown when `active: true` and MANIFOLD vetoes the observed task. */
export class ShadowVetoError extends Error {
  readonly riskScore: number;
  readonly suggestedAction: string | undefined;

  constructor(reason: string, riskScore: number, suggestedAction?: string) {
    super(`[MANIFOLD shadow] Vetoed: ${reason} (risk=${riskScore.toFixed(3)})`);
    this.name = "ShadowVetoError";
    this.riskScore = riskScore;
    this.suggestedAction = suggestedAction;
  }
}

// ---------------------------------------------------------------------------
// Internal audit record
// ---------------------------------------------------------------------------

interface _AuditRecord {
  task: BrainTask;
  manifoldVetoed: boolean;
  agentRan: boolean;
  riskScore: number;
  suggestedAction: string | undefined;
  timestamp: number;
}

// ---------------------------------------------------------------------------
// ShadowModeWrapper
// ---------------------------------------------------------------------------

/**
 * Non-destructive shadow-mode observer for Node.js AI agents.
 *
 * Wraps any async function and records what MANIFOLD *would* have decided,
 * without altering the agent's behaviour (unless `active: true`).
 */
export class ShadowModeWrapper {
  private readonly client: ManifoldClient;
  private readonly active: boolean;
  private readonly failOpen: boolean;
  private readonly _records: _AuditRecord[] = [];

  constructor(client: ManifoldClient, options: ShadowModeOptions = {}) {
    this.client = client;
    this.active = options.active ?? false;
    this.failOpen = options.failOpen ?? true;
  }

  /**
   * Observe a single agent call through MANIFOLD shadow mode.
   *
   * @param fn - The async agent function to (possibly) wrap.
   * @param task - A {@link BrainTask} describing what the agent is about to do.
   *
   * @returns The return value of `fn` (always, when `active: false`).
   *
   * @throws {ShadowVetoError} Only when `active: true` and MANIFOLD vetoes.
   */
  async observe<T>(fn: () => Promise<T>, task: BrainTask): Promise<T> {
    let manifoldVetoed = false;
    let riskScore = 0;
    let suggestedAction: string | undefined;

    try {
      const result = await this.client.evaluateShield(task);
      manifoldVetoed = result.vetoed;
      riskScore = result.risk_score;
      suggestedAction = result.suggested_action;

      if (this.active && manifoldVetoed) {
        this._records.push({
          task,
          manifoldVetoed,
          agentRan: false,
          riskScore,
          suggestedAction,
          timestamp: Date.now(),
        });
        throw new ShadowVetoError(result.reason, riskScore, suggestedAction);
      }
    } catch (err) {
      if (err instanceof ShadowVetoError) throw err;
      // MANIFOLD API unreachable or returned an error
      if (!this.failOpen) {
        throw new Error(
          `[MANIFOLD shadow] API error: ${(err as Error).message}`,
        );
      }
      // failOpen — log and let the agent run regardless
      console.warn("[MANIFOLD shadow] API error (failOpen=true):", (err as Error).message);
    }

    // In shadow mode (active=false) the agent ALWAYS runs
    const agentResult = await fn();

    this._records.push({
      task,
      manifoldVetoed,
      agentRan: true,
      riskScore,
      suggestedAction,
      timestamp: Date.now(),
    });

    return agentResult;
  }

  /**
   * Wrap an async agent function so that every call is automatically observed.
   *
   * @param fn - The async function to wrap.
   * @param taskBuilder - Maps the call arguments to a {@link BrainTask}.
   *
   * @returns The wrapped function with the same signature.
   */
  wrap<TArgs extends unknown[], TReturn>(
    fn: (...args: TArgs) => Promise<TReturn>,
    taskBuilder: (...args: TArgs) => BrainTask,
  ): (...args: TArgs) => Promise<TReturn> {
    return (...args: TArgs): Promise<TReturn> => {
      return this.observe(() => fn(...args), taskBuilder(...args));
    };
  }

  /**
   * Return the accumulated shadow-mode audit report.
   *
   * The report summarises all tasks observed since this wrapper was created
   * (or since the last {@link reset} call).
   */
  getReport(): ShadowReport {
    const total = this._records.length;
    if (total === 0) {
      return {
        total_tasks: 0,
        virtual_regret_saved: 0,
        hitl_escalations: 0,
        tool_failures_detected: 0,
        gossip_inoculation_speed: 0,
        adversarial_suspects: [],
        penalty_proposals: [],
        disagreement_rate: 0,
      };
    }

    const vetoes = this._records.filter((r) => r.manifoldVetoed);
    const escalations = vetoes.filter(
      (r) => r.suggestedAction === "escalate",
    ).length;
    const toolFailures = vetoes.filter(
      (r) => r.suggestedAction === "refuse" || r.suggestedAction === "verify",
    ).length;

    // Estimate regret saved: sum of risk scores for vetoed tasks
    const virtualRegretSaved = vetoes.reduce((acc, r) => acc + r.riskScore, 0);

    // Disagreement = fraction of tasks where MANIFOLD wanted to veto but didn't
    // (i.e. shadow mode let the agent run despite a veto recommendation)
    const shadowDisagreements = this.active
      ? 0
      : vetoes.filter((r) => r.agentRan).length;
    const disagreementRate = total > 0 ? shadowDisagreements / total : 0;

    // Gossip inoculation speed — estimated from veto rate over time
    let gossipSpeed = 0;
    if (this._records.length >= 2) {
      const first = this._records[0].timestamp;
      const last = this._records[this._records.length - 1].timestamp;
      const elapsedSec = Math.max((last - first) / 1000, 1);
      gossipSpeed = vetoes.length / elapsedSec;
    }

    // High-risk suspects (risk > 0.7 and vetoed)
    const suspects: AdversarialSuspect[] = vetoes
      .filter((r) => r.riskScore > 0.7)
      .map((r) => ({
        agent_id: r.task.domain ?? "unknown",
        deviation_score: r.riskScore,
        reason: `High risk score: ${r.riskScore.toFixed(3)}`,
      }));

    const proposals: PenaltyProposal[] = [];

    return {
      total_tasks: total,
      virtual_regret_saved: virtualRegretSaved,
      hitl_escalations: escalations,
      tool_failures_detected: toolFailures,
      gossip_inoculation_speed: gossipSpeed,
      adversarial_suspects: suspects,
      penalty_proposals: proposals,
      disagreement_rate: disagreementRate,
    };
  }

  /** Reset the audit log. */
  reset(): void {
    this._records.length = 0;
  }

  /** Total tasks observed. */
  get totalObserved(): number {
    return this._records.length;
  }
}
