# Development AI

A browser-extension-first AI assistant project.

## Current status
Version 0.2.0: extension popup chat workspace and local conversation storage are implemented. AI provider connectivity is not configured, so messages are stored locally and are not sent to a model.

## Load the extension locally
1. Open `chrome://extensions` in Chrome.
2. Enable **Developer mode**.
3. Select **Load unpacked**.
4. Choose this repository folder.

## Project structure
- `manifest.json` — Chrome Manifest V3 configuration.
- `popup/` — chat interface.
- `background/` — service worker and message handling.
- `docs/ARCHITECTURE.md` — system architecture and security boundaries.
- `docs/PROJECT_SCOPE.md` — current scope and exclusions.

## Security
Provider API keys must be kept on a backend, never in extension source code. Browser permissions should be added only when a specific feature requires them.
