/**
 * `shield` — TypeScript decorator factory that wraps an async function with a
 * MANIFOLD trust check before execution.
 *
 * Equivalent to the Python `@shield` decorator from `manifold/interceptor.py`.
 *
 * Usage
 * -----
 * ```ts
 * import { createShield } from "@manifold/client";
 *
 * const client = new ManifoldClient({ baseUrl: "http://localhost:8080" });
 * const shield = createShield(client);
 *
 * // Wrap any async handler
 * const safeTransfer = shield(
 *   async (amount: number, to: string) => {
 *     // ... actual transfer logic
 *   },
 *   (amount, to) => ({
 *     prompt: `Transfer $${amount} to ${to}`,
 *     domain: "finance",
 *     stakes: 0.95,
 *   })
 * );
 *
 * await safeTransfer(500, "acct-xyz"); // MANIFOLD vetoes high-risk calls
 * ```
 */

import type { BrainTask, InterceptResult } from "./types.js";
import { ManifoldClient, ManifoldApiError } from "./client.js";

/** Thrown when the MANIFOLD ActiveInterceptor vetoes an operation. */
export class ShieldVetoError extends Error {
  readonly result: InterceptResult;

  constructor(result: InterceptResult) {
    super(`[MANIFOLD] Operation vetoed: ${result.reason} (risk=${result.risk_score.toFixed(3)})`);
    this.name = "ShieldVetoError";
    this.result = result;
  }
}

/**
 * Options for the `shield` higher-order function.
 */
export interface ShieldOptions {
  /**
   * If `true`, a failed call to the MANIFOLD API (e.g. server unreachable)
   * will allow the wrapped function to proceed rather than throw.
   *
   * @default false
   */
  failOpen?: boolean;
}

/**
 * Create a `shield` wrapper factory bound to a {@link ManifoldClient}.
 *
 * @param client - An initialised {@link ManifoldClient}.
 * @param defaultOptions - Default {@link ShieldOptions} applied to every
 *   wrapped function.
 *
 * @returns A factory that wraps async functions with MANIFOLD interception.
 */
export function createShield(
  client: ManifoldClient,
  defaultOptions: ShieldOptions = {},
) {
  /**
   * Wrap `fn` with a MANIFOLD shield check.
   *
   * @param fn - The async function to protect.
   * @param taskBuilder - Maps the arguments of `fn` to a {@link BrainTask}
   *   that describes the operation for the interceptor.
   * @param options - Per-call overrides for {@link ShieldOptions}.
   */
  return function shield<TArgs extends unknown[], TReturn>(
    fn: (...args: TArgs) => Promise<TReturn>,
    taskBuilder: (...args: TArgs) => BrainTask,
    options: ShieldOptions = {},
  ): (...args: TArgs) => Promise<TReturn> {
    const failOpen = options.failOpen ?? defaultOptions.failOpen ?? false;

    return async (...args: TArgs): Promise<TReturn> => {
      const task = taskBuilder(...args);

      try {
        const result = await client.evaluateShield(task);
        if (result.vetoed) {
          throw new ShieldVetoError(result);
        }
      } catch (err) {
        if (err instanceof ShieldVetoError) throw err;
        // MANIFOLD API unreachable
        if (!failOpen) {
          throw new Error(
            `[MANIFOLD] Shield check failed: ${(err as Error).message}. ` +
              "Set failOpen: true to allow calls when the server is unavailable.",
          );
        }
        // failOpen — log and proceed
        console.warn("[MANIFOLD] Shield unreachable, proceeding (failOpen=true):", (err as Error).message);
      }

      return fn(...args);
    };
  };
}
