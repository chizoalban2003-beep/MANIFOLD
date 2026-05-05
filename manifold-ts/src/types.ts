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
  /** How relevant an external tool call is [0, 1]. */
  tool_relevance?: number;
  /** Urgency multiplier [0, 1]. */
  urgency?: number;
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
