# MANIFOLD AI Governance for VS Code

Govern every AI completion before it touches your code.

## Setup

1. Install extension
2. Open Settings → search "MANIFOLD"
3. Set `manifold.serverUrl` to your MANIFOLD server (e.g. `http://localhost:8080`)
4. Set `manifold.apiKey` to your API key from [/signup](/signup)

## Usage

- Select any code → right-click → **MANIFOLD: Check selected code for risk**
- Or: `Ctrl+Shift+P` → **MANIFOLD: Check selected code for risk**
- Enable `manifold.autoCheck` to check automatically on every save
- Open dashboard: Command Palette → **MANIFOLD: Open governance dashboard**

## What it does

MANIFOLD checks selected code or completions against your governance policy.

| Indicator | Meaning |
|-----------|---------|
| 🟢 Green | Low risk — permitted |
| 🟡 Yellow | Medium risk — verify first |
| 🔴 Red | Blocked by policy |

No code leaves your machine except to your own MANIFOLD server.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `manifold.serverUrl` | `http://localhost:8080` | MANIFOLD server URL |
| `manifold.apiKey` | *(empty)* | Your MANIFOLD API key |
| `manifold.enabled` | `true` | Enable governance checking |
| `manifold.autoCheck` | `false` | Auto-check on every save |

## Commands

| Command | Description |
|---------|-------------|
| `manifold.checkSelection` | Check selected code for risk |
| `manifold.openDashboard` | Open governance dashboard in browser |
