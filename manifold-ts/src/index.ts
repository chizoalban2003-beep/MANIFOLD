/**
 * @manifold/client — TypeScript client for the MANIFOLD Trust API.
 *
 * @example
 * ```ts
 * import { ManifoldClient, createShield } from "@manifold/client";
 *
 * const client = new ManifoldClient({ baseUrl: "http://localhost:8080" });
 * const shield = createShield(client);
 *
 * // Evaluate a task through @shield
 * const result = await client.evaluateShield({
 *   prompt: "Execute financial transaction",
 *   domain: "finance",
 *   stakes: 0.9,
 * });
 *
 * // Wrap any function with MANIFOLD governance
 * const safeOp = shield(
 *   async (amount: number) => processPayment(amount),
 *   (amount) => ({ prompt: `Payment of $${amount}`, domain: "finance", stakes: 0.9 }),
 * );
 * ```
 */

export { ManifoldClient, ManifoldApiError } from "./client.js";
export { createShield, ShieldVetoError } from "./shield.js";
export type {
  BrainTask,
  HandshakeResult,
  InterceptResult,
  InterceptorVeto,
  ManifoldClientOptions,
  ManifoldError,
  OrgPolicy,
  RecruitmentRequest,
  RecruitmentResult,
  ReputationScore,
} from "./types.js";
export type { ShieldOptions } from "./shield.js";
