/**
 * @manifold/client — TypeScript client for the MANIFOLD Trust API.
 *
 * @example
 * ```ts
 * import { ManifoldClient, createShield, ShadowModeWrapper } from "@manifold/client";
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
 *
 * // Non-destructive shadow audit
 * const shadow = new ShadowModeWrapper(client, { active: false });
 * const result2 = await shadow.observe(() => myAgent.run(prompt), {
 *   prompt,
 *   domain: "finance",
 *   stakes: 0.8,
 * });
 * const report = shadow.getReport();
 * ```
 */

export { ManifoldClient, ManifoldApiError } from "./client.js";
export { createShield, ShieldVetoError } from "./shield.js";
export { ShadowModeWrapper, ShadowVetoError } from "./shadow.js";
export type {
  BrainTask,
  BrainAction,
  BrainResponse,
  ToolProfile,
  ShadowReport,
  AdversarialSuspect,
  PenaltyProposal,
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
export type { ShadowModeOptions } from "./shadow.js";
export type {
  TrustTier,
  ToolRegistration,
  TrustSignal,
  AgentTrustScore,
} from "./types.js";

