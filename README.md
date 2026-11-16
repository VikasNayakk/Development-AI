# Development AI

A browser-extension-first AI assistant project.

## Current status
Version 0.3.2: local chat workspace, conversation persistence, Node.js AI gateway and automated backend integration tests are implemented. A real provider API key and model configuration are still required for live AI responses.

## Run the backend
1. Install Node.js 20.6 or newer.
2. Open a terminal in `backend/`.
3. Copy `.env.example` to `.env`.
4. Set `AI_BASE_URL`, `AI_API_KEY` and `AI_MODEL`.
5. Load the extension in Chrome and copy its extension ID.
6. Set `ALLOWED_EXTENSION_ORIGINS=chrome-extension://YOUR_EXTENSION_ID` in `.env`.
7. Run `npm start` from `backend/`.

The backend defaults to `http://127.0.0.1:8787`.

## Test the backend
From `backend/`, run:
- `npm run check` — syntax check.
- `npm test` — automated integration tests using a mock AI provider; no paid API key is needed.

## Load the extension
1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose this repository folder.
4. Start the backend before sending chat messages.

## Security
Provider API keys belong only in the backend environment file. Never commit `.env` or put keys in extension code. The backend binds to loopback by default and checks browser origins against the configured extension allowlist. It is not hardened for public deployment.
