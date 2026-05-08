# MANIFOLD World

An isometric game world visualising the MANIFOLD AI governance platform in real time.

## Features

- **16×16 isometric grid** with domain zones (Finance, DevOps, Healthcare, Legal)
- **6 AI agents** walking their zones, executing tasks, governed by MANIFOLD
- **The MANIFOLD Tower** at the grid centre — governs all agents
- **Resource nodes**, memory crystals, defence sensors, task pillars
- **Real-time WebSocket** connection to the MANIFOLD server
- **PWA** — installable on mobile as a fullscreen app
- **Touch controls** — pan (drag), zoom (pinch), tap to interact

## Running

Start the MANIFOLD server:

```bash
MANIFOLD_API_KEY=your_key python -m manifold.server --port 8080
```

Open in browser: `http://localhost:8080/world`

## Controls

| Action | Desktop | Mobile |
|--------|---------|--------|
| Pan | Mouse drag | One-finger drag |
| Zoom | Mouse wheel | Two-finger pinch |
| Interact | Left click | Tap |

## API Key

On first load you'll be prompted for your MANIFOLD API key and server URL.
These are stored in `localStorage` and used for WebSocket auth and task submission.
Without them the world runs in **demo mode** with simulated agent data.
