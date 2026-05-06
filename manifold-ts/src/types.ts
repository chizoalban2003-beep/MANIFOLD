/**
 * MANIFOLD TypeScript Client — types generated from the OpenAPI 3.0 specification
 * produced by Phase 23 (manifold/polyglot.py).
 *
 * These interfaces mirror the Python dataclasses exactly so that any payload
 * serialised by the Python server is valid here without transformation.
 */

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/** A task submitted to a MANIFOLD brain for evaluation and execution. */
export interface BrainTask {
  /** Human-readable task description. */
  prompt: string;
  /** Domain category, e.g. "finance", "medical", "legal". */
  domain?: string;
  /** Cognitive complexity estimate [0, 1]. */
  complexity?: number;
  /** Stakes of the task — how bad is failure? [0, 1]. */
  stakes?: number;
  /** Uncertainty about the best course of action [0, 1]. */
  uncertainty?: number;
  /** Confidence in the information source [0, 1]. */
  source_confidence?: number;
  /** How relevant an external tool call is [0, 1]. */
  tool_relevance?: number;
  /** Time pressure on the decision [0, 1]. */
  time_pressure?: number;
  /** Safety sensitivity of the domain [0, 1]. */
  safety_sensitivity?: number;
  /** Value of collaborating with another agent [0, 1]. */
  collaboration_value?: number;
  /** Urgency multiplier [0, 1]. */
  urgency?: number;
  /** User patience for long-running tasks [0, 1]. */
  user_patience?: number;
  /** Whether the goal can change mid-execution. */
  dynamic_goal?: boolean;
  /** Relative weight for batch scheduling [0, 1]. */
  weight?: number;
}

/**
 * All possible actions the MANIFOLD brain can recommend.
 * Mirrors the Python `BrainAction` type in `manifold/brain.py`.
 */
export type BrainAction =
  | "answer"
  | "clarify"
  | "retrieve"
  | "verify"
  | "use_tool"
  | "delegate"
  | "plan"
  | "explore"
  | "exploit"
  | "wait"
  | "escalate"
  | "refuse"
  | "stop";

/** Full brain decision response (extended from InterceptResult). */
export interface BrainResponse {
  /** Recommended action. */
  action: BrainAction;
  /** Confidence in the decision [0, 1]. */
  confidence: number;
  /** Estimated execution cost [0, 1]. */
  cost_estimate: number;
  /** Estimated risk score [0, 1]. */
  risk_estimate: number;
  /** Optional human-readable reasoning. */
  reasoning?: string;
}

/** Profile describing a tool registered with the ConnectorRegistry. */
export interface ToolProfile {
  /** Tool identifier. */
  name: string;
  /** Cost per call (normalised [0, 1]). */
  cost: number;
  /** Typical latency in milliseconds. */
  latency: number;
  /** Observed reliability score [0, 1]. */
  reliability: number;
  /** Risk introduced by using this tool [0, 1]. */
  risk: number;
  /** Value (asset) produced per successful call [0, 1]. */
  asset: number;
}

/** Adversarial agent suspected of gaming the reputation system. */
export interface AdversarialSuspect {
  /** Agent or tool identifier. */
  agent_id: string;
  /** Nash equilibrium deviation score. */
  deviation_score: number;
  /** Reason the agent was flagged. */
  reason: string;
}

/** Penalty rule change proposed by the PenaltyOptimizer. */
export interface PenaltyProposal {
  /** Rule name to change. */
  rule_name: string;
  /** Current penalty value. */
  old_value: number;
  /** Proposed new penalty value. */
  new_value: number;
  /** Confidence in the proposal [0, 1]. */
  confidence: number;
  /** Explanation of why the change is proposed. */
  rationale: string;
}

/** Aggregated shadow-mode audit report. */
export interface ShadowReport {
  /** Total tasks observed in shadow mode. */
  total_tasks: number;
  /** Estimated regret saved by MANIFOLD interventions. */
  virtual_regret_saved: number;
  /** Number of tasks escalated to a human reviewer. */
  hitl_escalations: number;
  /** Number of tool failures detected. */
  tool_failures_detected: number;
  /** Speed (tasks/sec) at which gossip propagated inoculation data. */
  gossip_inoculation_speed: number;
  /** Agents suspected of adversarial behaviour. */
  adversarial_suspects: AdversarialSuspect[];
  /** Penalty update proposals from the PenaltyOptimizer. */
  penalty_proposals: PenaltyProposal[];
  /** Fraction of tasks where MANIFOLD disagreed with the wrapped agent [0, 1]. */
  disagreement_rate: number;
}

/** Result from the ActiveInterceptor (@shield) endpoint. */
export interface InterceptResult {
  /** Whether the task was vetoed. */
  vetoed: boolean;
  /** Reason for the decision. */
  reason: string;
  /** Risk score computed by the brain [0, 1]. */
  risk_score: number;
  /** Confidence in the decision [0, 1]. */
  confidence: number;
  /** Suggested action when vetoed. */
  suggested_action?: string;
}

/** Returned when the interceptor blocks a task (HTTP 422). */
export interface InterceptorVeto {
  /** Always true. */
  vetoed: true;
  /** Human-readable veto reason. */
  reason: string;
  /** Risk score that triggered the veto [0, 1]. */
  risk_score: number;
}

/** Lightweight snapshot of a remote organisation's published policy. */
export interface OrgPolicy {
  /** Unique organisation identifier. */
  org_id: string;
  /** Minimum tool reliability this org guarantees [0, 1]. */
  min_reliability?: number;
  /** Maximum risk score any call from this org may produce [0, 1]. */
  max_risk?: number;
  /** Primary operational domain. */
  domain?: string;
  /** Policy version string. */
  version?: string;
  /** Free-text notes. */
  notes?: string;
}

/** Result of a B2B policy handshake. */
export interface HandshakeResult {
  /** Whether the two org policies are compatible. */
  compatible: boolean;
  /** Human-readable explanation. */
  reason: string;
  /** Negotiated trust score for this session [0, 1]. */
  trust_score: number;
}

/** Live reputation score for a tool or org. */
export interface ReputationScore {
  /** Tool or organisation identifier. */
  agent_id: string;
  /** Reliability score over the sliding window [0, 1]. */
  reliability: number;
  /** Number of observations in the window. */
  sample_count: number;
  /** ISO-8601 timestamp of the last update. */
  last_updated: string;
}

/** Request to the Sovereign Recruiter. */
export interface RecruitmentRequest {
  /** Task description. */
  task_description: string;
  /** Maximum acceptable cost per call. */
  max_cost?: number;
  /** Required minimum reliability [0, 1]. */
  min_reliability?: number;
  /** Required tool capability tags. */
  required_capabilities?: string[];
}

/** Outcome from the Sovereign Recruiter. */
export interface RecruitmentResult {
  /** Whether a suitable tool was recruited. */
  recruited: boolean;
  /** Recruited tool identifier (if recruited). */
  tool_id?: string;
  /** Recruitment score [0, 1]. */
  score: number;
  /** Human-readable explanation. */
  reason: string;
}

/** Standard API error envelope. */
export interface ManifoldError {
  /** HTTP status code. */
  code: number;
  /** Human-readable message. */
  message: string;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** Options accepted by ManifoldClient. */
export interface ManifoldClientOptions {
  /**
   * Base URL of the MANIFOLD API server.
   * @default "http://localhost:8080"
   */
  baseUrl?: string;
  /**
   * Optional Bearer token for authenticated deployments.
   */
  apiKey?: string;
  /**
   * Timeout for each request in milliseconds.
   * @default 10000
   */
  timeoutMs?: number;
}

