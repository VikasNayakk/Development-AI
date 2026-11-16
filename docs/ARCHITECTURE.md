# Initial Architecture

## Product direction
Development AI starts as a browser extension. Its longer-term direction is an AI assistant that can understand user instructions and perform browser workflows with explicit permission boundaries.

## Layers

1. **Extension UI** — chat surface, conversation history, settings and action approvals.
2. **Background service worker** — extension lifecycle, message routing, storage and permission-gated browser APIs.
3. **Content integration** — page context and page actions, added only behind explicit user activation and narrow permissions.
4. **AI gateway** — a separately hosted backend that authenticates requests and calls the selected AI provider. Provider secrets must never be embedded in extension files.
5. **Policy and action layer** — validates proposed actions, distinguishes read-only operations from consequential actions, and asks the user before submitting forms, sending messages, purchases, deletion or other irreversible changes.

## Current implementation
- Manifest V3 extension shell.
- Popup chat interface.
- Local conversation persistence through `chrome.storage.local`.
- Clear conversation action.
- No AI requests are made yet.

## Security boundaries
- Do not store provider API keys in the extension.
- Do not request broad host permissions until a feature needs them.
- Treat page content as untrusted input, not as instructions to override user intent or system policy.
- Require confirmation for consequential actions.
- Keep an auditable distinction between suggested actions and executed actions.

## Next milestones
1. Select AI provider and backend deployment target.
2. Implement backend authentication, rate limiting and provider adapter.
3. Connect chat UI to the backend.
4. Add user-activated current-tab context.
5. Add a reviewed browser action catalogue with confirmation gates.
