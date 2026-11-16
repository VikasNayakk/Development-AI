# Development AI

A browser-extension-first AI assistant project.

## Current status
Version 0.3.1: local chat workspace, conversation persistence and a Node.js AI gateway are implemented. A provider API key, model configuration and extension-origin allowlist are required before AI replies can work.

## Run the backend
1. Install Node.js 20.6 or newer.
2. Open a terminal in `backend/`.
3. Copy `.env.example` to `.env`.
4. Set `AI_BASE_URL`, `AI_API_KEY` and `AI_MODEL`.
5. Load the extension in Chrome, open `chrome://extensions`, and copy its extension ID.
6. Add the exact origin `chrome-extension://YOUR_EXTENSION_ID` to `ALLOWED_EXTENSION_ORIGINS` in `.env`.
7. Run `npm start` from `backend/`.

The backend defaults to `http://127.0.0.1:8787`.

## Load the extension
1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose this repository folder.
4. Start the backend before sending chat messages.

## Project structure
- `manifest.json` — Chrome Manifest V3 configuration.
- `popup/` — chat interface.
- `background/` — service worker and local message storage.
- `backend/` — provider-neutral Node.js gateway.
- `docs/` — architecture and project scope.

## Security
Provider API keys belong only in the backend environment file. Never commit `.env` or put keys in extension code. The backend is local-development scaffolding and is not hardened for public deployment.
