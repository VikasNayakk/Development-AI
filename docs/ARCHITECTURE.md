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
- Extension sends chat history to the local backend.
- Backend checks browser origins against an explicit allowlist.
- Automated integration tests use a mock provider and require no external API key.

## AI provider configuration
The backend accepts an OpenAI-compatible Chat Completions endpoint. Set `AI_BASE_URL`, `AI_API_KEY` and `AI_MODEL` in `backend/.env`. Provider credentials must never be put in extension files or committed to Git.

## Local development
1. Install Node.js 20.6 or newer.
2. In `backend/`, copy `.env.example` to `.env`.
3. Set the provider endpoint, API key and model in `.env`.
4. Load the extension in Chrome and copy its extension ID.
5. Set the exact extension origin in `ALLOWED_EXTENSION_ORIGINS`.
6. Run `npm start` in `backend/`.

## Testing
Run `npm run check` and `npm test` from `backend/`. Integration tests start a mock provider and verify health, successful chat, input validation and origin rejection.

## Security boundaries
- The backend binds to loopback (127.0.0.1) by default.
- Requests with browser origins are accepted only when the exact origin is allowlisted.
- Do not expose the local gateway to a public network without authentication, rate limiting and TLS.
- Do not store provider keys in extension source.
- Treat page content as untrusted data.
- Ask the user before consequential browser actions.
