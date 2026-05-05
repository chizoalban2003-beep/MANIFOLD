# @manifold/client

> **TypeScript client for the MANIFOLD Trust API** — the Trust Operating System for AI agents.

Generated from the Phase 23 OpenAPI 3.0 specification (`manifold/polyglot.py`). Zero runtime dependencies; uses the built-in `fetch` API (Node ≥ 18).

## Installation

```bash
npm install @manifold/client
```

## Quick start

```typescript
import { ManifoldClient, createShield } from "@manifold/client";

// 1. Create a client pointed at your Python-hosted MANIFOLD server
const client = new ManifoldClient({ baseUrl: "http://localhost:8080" });

// 2. Evaluate a task through the @shield interceptor
const result = await client.evaluateShield({
  prompt: "Transfer $50,000 to account X",
  domain: "finance",
  stakes: 0.95,
  complexity: 0.6,
});

if (result.vetoed) {
  console.error("MANIFOLD blocked the operation:", result.reason);
} else {
  console.log(`Risk score: ${result.risk_score} — proceeding`);
}
```

## The `shield` higher-order function

The TypeScript equivalent of the Python `@shield` decorator:

```typescript
import { ManifoldClient, createShield, ShieldVetoError } from "@manifold/client";

const client = new ManifoldClient({ baseUrl: "http://localhost:8080" });
const shield = createShield(client);

// Wrap any async function — MANIFOLD evaluates the task before execution
const safeTransfer = shield(
  async (amount: number, to: string) => {
    // your actual transfer logic
    return { txId: "abc123" };
  },
  // taskBuilder maps arguments → BrainTask for the interceptor
  (amount, to) => ({
    prompt: `Transfer $${amount} to account ${to}`,
    domain: "finance",
    stakes: amount > 10_000 ? 0.95 : 0.5,
  }),
);

try {
  const tx = await safeTransfer(50_000, "acct-xyz");
  console.log("Transfer complete:", tx.txId);
} catch (err) {
  if (err instanceof ShieldVetoError) {
    console.error("Blocked by MANIFOLD:", err.result.reason);
  }
}
```

### `failOpen` mode (for non-critical paths)

```typescript
const shield = createShield(client, { failOpen: true });
// If the MANIFOLD server is unreachable, the wrapped function proceeds anyway
```

## API reference

### `ManifoldClient`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `evaluateShield(task)` | `POST /shield` | Run a BrainTask through the ActiveInterceptor |
| `b2bHandshake(policy)` | `POST /b2b/handshake` | Perform a cross-org policy handshake |
| `getReputation(agentId)` | `GET /reputation/:id` | Query live reliability from the ReputationHub |
| `recruit(request)` | `POST /recruit` | Hire the best-fit tool via SovereignRecruiter |
| `getPolicy()` | `GET /policy` | Retrieve the local org's published OrgPolicy |

### `ManifoldClientOptions`

```typescript
interface ManifoldClientOptions {
  baseUrl?: string;   // default: "http://localhost:8080"
  apiKey?: string;    // Bearer token for authenticated deployments
  timeoutMs?: number; // default: 10000
}
```

### Error handling

```typescript
import { ManifoldApiError } from "@manifold/client";

try {
  await client.b2bHandshake({ org_id: "unknown-org" });
} catch (err) {
  if (err instanceof ManifoldApiError) {
    console.error(`HTTP ${err.status}: ${err.body.message}`);
  }
}
```

## Integration examples

### With the Vercel AI SDK

```typescript
import { generateText } from "ai";
import { openai } from "@ai-sdk/openai";
import { ManifoldClient, createShield } from "@manifold/client";

const manifold = new ManifoldClient();
const shield = createShield(manifold);

const safeGenerate = shield(
  (prompt: string) => generateText({ model: openai("gpt-4o"), prompt }),
  (prompt) => ({ prompt, domain: "general", complexity: 0.5, stakes: 0.3 }),
);

const { text } = await safeGenerate("Summarise the latest earnings report");
```

### With LangChain.js

```typescript
import { ChatOpenAI } from "@langchain/openai";
import { ManifoldClient, createShield } from "@manifold/client";

const manifold = new ManifoldClient();
const shield = createShield(manifold);
const llm = new ChatOpenAI({ model: "gpt-4o" });

const safeInvoke = shield(
  (input: string) => llm.invoke(input),
  (input) => ({ prompt: input, domain: "general", stakes: 0.4 }),
);
```

## Running the MANIFOLD server

The TypeScript client communicates with the Python-hosted REST server. Start it with:

```bash
# From the MANIFOLD repo root
pip install manifold-ai
# Your FastAPI/Flask server wrapping the manifold package
uvicorn server:app --port 8080
```

## License

MIT — see the [MANIFOLD repository](https://github.com/chizoalban2003-beep/MANIFOLD) for details.
