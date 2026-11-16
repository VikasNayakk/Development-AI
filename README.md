# Development AI

A browser-extension-first AI assistant powered by a local model through Ollama.

## Current status
Version 0.4.0: extension chat, local conversation storage, Node.js gateway, Ollama integration, model availability status and backend integration tests are implemented. The Ollama runtime and model must be installed on the computer before live local inference works.

## Install local AI runtime
1. Install Ollama for your operating system from https://ollama.com/download.
2. Start Ollama.
3. In a terminal, download a model:
   `ollama pull qwen2.5:7b`
4. Confirm the model is available:
   `ollama list`

The default model is `qwen2.5:7b`. Model memory requirements depend on quantization and hardware. If the computer has limited RAM/VRAM, select a smaller model and set the same model tag in `backend/.env`.

## Configure and start the backend
1. Install Node.js 20.6 or newer.
2. Open a terminal in `backend/`.
3. Copy `.env.example` to `.env`.
4. Set `ALLOWED_EXTENSION_ORIGINS=chrome-extension://YOUR_EXTENSION_ID` after loading the extension.
5. Keep the default Ollama settings unless Ollama uses a different local port/model.
6. Run `npm start`.

The backend defaults to `http://127.0.0.1:8787`; Ollama defaults to `http://127.0.0.1:11434`. No cloud API key is required for local Ollama inference.

## Load the extension
1. Open `chrome://extensions`.
2. Enable **Developer mode**.
3. Select **Load unpacked** and choose this repository folder.
4. Copy the extension ID and add its exact origin to `backend/.env`.
5. Start Ollama, then the backend, and open the extension popup.

## Test
From `backend/`, run `npm run check` and `npm test`. The integration tests use a mock Ollama-compatible local server and do not download a model or call an external AI service.

## Privacy and security
Prompts are sent to the local Ollama service through the local backend. No cloud AI provider is used by the default configuration. Provider/model software may have its own license and data-handling terms. The backend is local-development scaffolding and is not hardened for public network exposure.
