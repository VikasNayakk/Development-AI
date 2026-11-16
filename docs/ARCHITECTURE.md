# Architecture

## Layers
1. **Extension UI** — chat surface, conversation history and user controls.
2. **Background service worker** — message routing and local storage.
3. **Local AI gateway** — Node.js backend that validates requests and forwards them to Ollama.
4. **Local model runtime** — Ollama serves the selected model on the local machine.
5. **Policy and action layer** — future browser actions with explicit permission and confirmation gates.

## Current implementation
- Manifest V3 extension and popup chat.
- Local conversation persistence.
- Node.js backend with `GET /health`, `GET /v1/status` and `POST /v1/chat`.
- Default provider: Ollama at `http://127.0.0.1:11434`.
- Default model tag: `qwen2.5:7b`.
- Backend probes Ollama's model list to report whether the configured model is installed.
- Automated tests use a mock Ollama-compatible HTTP server.

## Local model setup
Install Ollama, pull the selected model (default `ollama pull qwen2.5:7b`), copy `backend/.env.example` to `backend/.env`, set the extension origin allowlist, and start the backend. No cloud API key is required for Ollama.

## Optional alternative provider
An OpenAI-compatible endpoint can be selected with `AI_PROVIDER=openai-compatible`, `AI_BASE_URL`, `AI_API_KEY` and `AI_MODEL`.

## Security boundaries
- Backend binds to loopback (127.0.0.1) by default.
- Browser origins must match the explicit allowlist.
- Provider keys, when using a remote compatible provider, stay in backend environment variables.
- Treat page content as untrusted data.
- Ask the user before consequential browser actions.
- Do not expose the local gateway publicly without authentication, rate limiting and TLS.
