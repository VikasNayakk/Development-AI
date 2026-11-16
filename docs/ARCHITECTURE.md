# Architecture

## Layers
1. **Extension UI** — chat surface, conversation history and user controls.
2. **Background service worker** — message routing and local storage.
3. **AI gateway** — local Node.js backend that authenticates requests and calls an OpenAI-compatible provider.
4. **Policy and action layer** — future browser actions with explicit permission and confirmation gates.

## Current implementation
- Manifest V3 extension.
- Popup chat interface and local conversation persistence.
- Node.js backend with `GET /health` and `POST /v1/chat`.
- Provider credentials are read from backend environment variables.
- The extension sends conversation messages to the local backend.

## AI provider configuration
The backend accepts an OpenAI-compatible Chat Completions endpoint. Set `AI_BASE_URL`, `AI_API_KEY` and `AI_MODEL` in `backend/.env`. Provider credentials must never be put in extension files or committed to Git.

## Local development
1. Install Node.js 20 or newer.
2. In `backend/`, copy `.env.example` to `.env`.
3. Set the provider endpoint, API key and model in `.env`.
4. Start the backend with `npm start`.
5. Load the repository folder as an unpacked Chrome extension.

## Security boundaries
- The backend binds to loopback (127.0.0.1) by default.
- Configure the exact extension origin in `ALLOWED_EXTENSION_ORIGINS` when required by browser CORS.
- Do not expose the local gateway to a public network without authentication, rate limiting and TLS.
- Do not store provider keys in extension source.
- Treat page content as untrusted data.
- Ask the user before consequential browser actions.
