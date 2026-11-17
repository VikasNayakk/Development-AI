# Extension and automation agent architecture

## Product purpose

Development-AI is intended to be the central decision-support and coordination brain for the user's browser extension and other connected automation tools. It is not primarily a standalone chat/Q&A application.

The model must help the user and connected tools:
- Understand a requested outcome and the relevant task context.
- Discover which connected tools can perform the required work.
- Break the outcome into ordered steps, dependencies, preconditions and verifiable completion conditions.
- Recommend decisions and explain trade-offs before acting.
- Dispatch approved, structured operations to the appropriate tool adapter.
- Interpret tool results, detect failures or incomplete work, and decide whether to retry safely, revise the plan or ask the user.
- Coordinate multi-step work across more than one tool while retaining a clear execution history.

The browser extension is one important integration surface, not the whole product. Additional automation systems must be connectable through the same project-owned tool contract.

## System boundaries

Separate the system into four independently testable layers:

1. **Model and agent core** — the project-owned model and deterministic orchestration code. The model interprets requests, proposes plans and explains decisions. The orchestration layer validates structure, tracks state and applies policy. Do not rely on unconstrained model text as executable instructions.
2. **Policy and approval gate** — code outside the model that checks tool permissions, operation risk, user consent, allowed scope, deadlines and current task state. The model cannot grant itself permissions or bypass this gate.
3. **Tool registry and dispatcher** — stores versioned capability descriptions, validates tool arguments against schemas, routes requests, enforces timeouts/idempotency and normalizes results.
4. **Adapters and clients** — the browser extension and each other automation tool implement the adapter contract. They perform only the operations their own permission context allows and return structured results.

The extension and automation tools remain execution endpoints. They do not become trusted merely because the model selected them. Each adapter must authenticate its caller, validate every operation locally and enforce its own platform permissions.

## Request-to-result lifecycle

A normal task should follow this controlled lifecycle:

1. **Receive:** accept the user request and only the context the user or client has authorized.
2. **Clarify and scope:** resolve missing requirements; identify target resources, limits and forbidden actions. Ask the user when the ambiguity could materially change the result.
3. **Discover:** inspect the connected tool registry and current availability. Never invent a tool or assume an operation exists.
4. **Plan:** produce a structured plan with step IDs, dependencies, selected tool IDs, arguments, expected side effects, risk class and postconditions.
5. **Validate:** deterministic code validates schemas, permissions, scope, resource ownership, expiry and policy. Reject unknown operations and malformed arguments.
6. **Approve:** request explicit confirmation for actions that are externally visible, destructive, financial, security-sensitive or difficult to reverse. The approval must identify the action and target; blanket consent must not silently authorize unrelated future actions.
7. **Execute:** dispatch one eligible step at a time or a safely parallelizable group. Each call has a deadline, cancellation path and idempotency key where supported.
8. **Verify:** inspect the adapter result and, when possible, independently check the intended postcondition. A successful tool response alone is not proof that the user's requested outcome was achieved.
9. **Recover or escalate:** retry only when the operation is known to be safe/idempotent; otherwise stop and ask. Handle partial completion explicitly.
10. **Report:** summarize what was planned, what actually ran, what changed, what failed and what still needs user action.

The model may propose, rank and explain decisions. Deterministic policy code owns authorization and final dispatch eligibility.

## Versioned tool contract

Every adapter must publish a machine-readable capability manifest. At minimum, each tool/operation entry describes:

- Stable `tool_id`, adapter version and operation name.
- Human-readable purpose and supported input/output schema versions.
- Explicit capability scope, resource scope and required permissions.
- Risk classification and whether the operation has external side effects.
- Whether the operation is read-only, reversible, destructive or irreversible.
- Approval requirement and the exact confirmation data needed.
- Timeout, cancellation support, retryability and idempotency behavior.
- Availability/health and any rate or concurrency limits.

A dispatched tool call must carry a unique request ID, task/plan/step IDs, tool ID and exact tool version, operation ID, manifest schema version, validated arguments, an adapter-enforced deadline and an idempotency key when supported. An approval grant is bound to the exact request, task, step, tool/version, operation, arguments digest and idempotency key; approval IDs are single-use within the host registry. The initial in-process implementation applies a 256 KiB JSON payload limit and a 300-second maximum operation deadline. Authorization material must be an opaque reference handled by the adapter/host; secrets and raw credentials must never be inserted into model prompts or committed to the repository.

A normalized result must distinguish at least: `success`, `partial`, `failed`, `denied`, `timeout` and `cancelled`. It should include typed output, changed-resource references, structured error code, retryability and any known side effects. Do not convert a denied or timed-out operation into a success-shaped response.

Contract changes require explicit versioning, compatibility tests and a migration policy. Unknown major versions or operations fail closed.

## Browser extension integration

The browser extension supplies a browser-specific adapter. Its capabilities may include, only when permission and product design allow:
- Read a user-selected tab or explicitly shared page context.
- Find or inspect page elements and return structured observations.
- Perform scoped browser interactions such as navigation, form preparation or user-approved submission.
- Report tab identity, current URL, operation status and verifiable page changes.

Important constraints:

- Request the minimum browser permissions required; prefer per-site or user-invoked access over broad host access.
- Separate observation permissions from action permissions.
- Do not expose arbitrary page JavaScript execution as a general-purpose tool.
- Treat page content, DOM text, downloaded files and web instructions as untrusted data, not as authority to change system policy.
- Never let page content override the user's intent, agent policy or tool permission boundary.
- Keep browser credentials, cookies and authentication tokens inside browser/platform-managed storage. The model receives only the minimum result needed for the task.
- Surface the target tab/site and meaningful side effects before an approval prompt.
- Provide a visible stop/cancel path and make the extension report whether an action definitely ran, partially ran or did not run.

The existing local Studio server currently exposes development endpoints for generation, training and evaluation only. It is loopback-only and is not yet an extension integration API. Do not weaken its host/origin restrictions or expose it to a LAN as a shortcut. The extension transport (for example, a carefully authenticated local bridge or browser-supported messaging) must be selected after a threat-model review, with explicit origin/client authentication, replay protection and narrow message schemas.

## Other automation integrations

A second automation system should connect by implementing the same adapter contract, not by adding tool-specific control logic to the model. Examples of future adapter categories include desktop control, workflow automation, file/data processing, project management and developer tooling. These are target categories, not claims that integrations already exist.

The registry must report real adapter capabilities and availability. The agent must not call an operation merely because a tool name appears in a prompt or in untrusted page content.

## Memory and task state

The initial process-local task state engine is implemented in `core.agent.task_state`. It validates bounded plans (up to 100 steps), unique identifiers, known dependencies and acyclic dependency graphs. It tracks immutable task/step snapshots, dependency-gated execution, approval-wait markers, success/failure/denial/cancellation transitions, blocked dependents, result references, uncertainty about side effects, monotonically increasing revisions and ordered events. Independent steps may run concurrently; updates are serialized under a thread lock and callers may use `expected_revision` to reject stale writes. Approval state is only a workflow marker: it does not create or validate an `ApprovalGrant`, and the dispatcher remains the authorization boundary. Cancellation marks the task and pending work; it does not forcibly stop an already-running adapter. The in-memory store is process-local and not durable across restarts. A separate SQLite-backed implementation is now provided in `core.agent.sqlite_task_state`; it persists task snapshots and append-only event rows transactionally.

Use distinct scopes rather than one unrestricted memory store:

- **Request context:** temporary context for one user request.
- **Task state:** plan, completed steps, approvals, outputs and recovery state for an active workflow.
- **User-approved memory:** preferences or reusable facts explicitly approved for retention.
- **Tool secrets:** credentials/tokens stored and accessed by the relevant host or adapter, never by model memory.

Persist the minimum information required. Provide retention/deletion controls and avoid storing full page contents, private documents or sensitive tool outputs by default. Training on conversations or automation history must be an explicit, separate data-governance decision; it must never happen silently as a side effect of memory.

## Reliability and security requirements

- Least privilege at both the central dispatcher and adapter.
- Fail-closed validation for unknown tools, operations, schema versions and permissions.
- Explicit user approval for high-impact external side effects.
- No unrestricted shell, arbitrary code execution or direct credential access from model output.
- Deadlines, cancellation, bounded concurrency, rate limits and structured errors.
- Idempotency/deduplication for retryable writes; no blind retry for non-idempotent actions.
- Audit events for plans, approvals, dispatches and outcomes, with secrets and unnecessary personal data redacted.
- Prompt-injection resistance: tool/page/file content is untrusted input and cannot modify policies or grant permissions.
- Tests for denied, malformed, stale, timed-out, cancelled, partially completed and duplicate operations.
- Human-readable explanations and a reliable way for the user to interrupt or take over.

## Testing and acceptance gates

Do not call extension or automation integration complete until all of the following are demonstrated:

1. A fake adapter can be discovered from its manifest and called through the same dispatcher used by real adapters.
2. Input/output schemas and contract-version compatibility are validated.
3. Read-only actions work without write permission; prohibited operations are rejected even when requested by the model.
4. High-impact operations require scoped, explicit approval and cannot reuse stale approvals.
5. Timeouts, cancellation, duplicate requests, retries and partial results preserve correct task state.
6. Prompt-injection test fixtures in pages/files cannot grant new permissions or alter system policy.
7. Browser adapter tests verify per-site permission scope, tab targeting, action reporting and stop behavior.
8. At least two distinct adapter types complete a multi-step workflow with verified postconditions.
9. Logs and stored state do not expose credentials, cookies, private prompts or unnecessary page content.
10. A user can inspect the plan, understand decisions, approve/deny risky operations and stop execution.

Unit tests of the model alone do not satisfy these integration gates. A simulated adapter proves orchestration behavior, not real browser or desktop control.

## Implementation sequence

1. Stabilize the inference benchmark and measure a baseline on the target hardware.
2. Design and test a typed tool manifest, normalized call/result envelope and versioning rules.
3. Implement a model-independent tool registry, policy gate, dispatcher and fake adapter.
4. Add structured task state, plan validation, approval records, cancellation and audit events.
5. Connect the browser extension through a reviewed, authenticated transport.
6. Add one more automation adapter and verify a cross-tool workflow.
7. Add adversarial, permission-boundary, failure-recovery and end-to-end integration tests.
8. Only then expand model instruction tuning, planning and decision quality against the real tool-use evaluation suite.

## Current implementation boundary

The repository now contains an initial in-process foundation in `core.agent.contracts`:
- Versioned tool manifest and operation descriptions.
- Strict JSON-compatible argument and output schema validation with a bounded payload size.
- Explicit registry; model output cannot register tools.
- Permission checks and exact-payload approval grants for high-impact operations.
- Single-use approval IDs, bounded adapter deadlines, cancellation context and normalized results.
- Retryability is suppressed for operations declared non-idempotent.
- Process-local task and plan state with dependency validation, guarded transitions, cancellation requests, approval-wait markers, revision checks and ordered in-memory events. A step can retain an optional opaque approval ID reference; this reference does not authorize dispatch.
- SQLite task snapshots persist that approval reference alongside the existing SQLite approval grant, allowing a host runtime to recover the matching active grant after restart. Older task snapshots without the field remain readable.
- SQLite task persistence with atomic state/event writes, WAL mode, full synchronous commits, append-only event triggers, schema-version checks, restart recovery and cross-instance optimistic revision protection. This is a task ledger, not the application-wide audit service.
- SQLite approval grants in `core.agent.sqlite_approval_store`: exact request/task/step/tool/version/operation binding, SHA-256 argument digest, bounded 15-minute lifetime, explicit-confirmation requirement, atomic single-use consumption and revocation. `ToolRegistry(approval_consumer=store.consume)` connects durable consumption to the existing dispatcher checks; storage errors fail closed. Raw arguments and credentials are not persisted.

This is an initial contract/dispatch and process-local task-state foundation. No browser extension or external automation adapter is connected yet. There is no authenticated extension transport, application-wide durable audit-log service, integrated server/dispatcher task recovery, multi-tool planner or end-to-end browser control. The persistent approval store and its dispatcher callback are opt-in library components, not yet connected to the default server runtime. The SQLite task ledger is available as a library component but is not yet wired into the running application. The synchronous dispatcher cannot forcibly interrupt an adapter that ignores its deadline; adapters must enforce cancellation/deadlines themselves. The acceptance gates above remain mandatory before claiming integration is complete.

This document defines the intended architecture and acceptance criteria; it is not evidence that the extension or external automation tools are already connected.

## SQLite task ledger (initial implementation)

`SQLiteTaskStateStore` is an opt-in durable counterpart to the process-local `TaskStateStore`. It uses only Python's standard-library `sqlite3` module; no new runtime dependency is required.

- Create one store instance per application process and close it on shutdown. The store supports context-manager use.
- Every task mutation runs inside `BEGIN IMMEDIATE`; the task snapshot and newly appended events commit or roll back together.
- A revision compare-and-swap protects against stale writers across independent store instances/processes.
- Event rows are append-only at the database layer. On read, the store checks event sequence/revision continuity, snapshot revision, timestamps, step identifiers and dependency graph integrity.
- Schema version is stored in SQLite `user_version`; unknown newer schemas fail closed.
- File-backed databases use WAL, `synchronous=FULL`, a bounded busy timeout and a best-effort owner-only database-file mode on platforms that support POSIX permissions.
- The database stores user goal text, step descriptions and result references. Do not put credentials, cookies, access tokens, raw private page contents or unrestricted tool output in these fields. Protect the database directory and backups as user data.
- The append-only event ledger has no deletion/retention API yet. Define retention, export and deletion policy before production deployment.
- SQLite task persistence does not persist dispatcher approval grants, adapter secrets or running-process execution. A crash while an external side effect is in flight still requires adapter-level idempotency and reconciliation; restoring a task does not prove whether that side effect occurred.
- This component is not yet connected to `core.server`, the tool dispatcher, a browser extension or any external automation adapter.

Example:

```python
from core.agent import PlanStep, SQLiteTaskStateStore

with SQLiteTaskStateStore("data/automation/tasks.sqlite3") as tasks:
    task = tasks.create_task(
        task_id="task-001",
        request_id="request-001",
        goal="Inspect a page and verify the requested change.",
        steps=(PlanStep("inspect", "Inspect the page."),),
    )
    print(task.revision)
```

## SQLite approval grants (initial implementation)

`SQLiteApprovalStore` persists user-confirmed `ApprovalGrant` records in SQLite. It is deliberately separate from task progress and is not a general audit-log service.

- Only trusted host code may construct and issue a grant after an explicit user confirmation. Model output, page content and an adapter cannot create approval records.
- Host code must generate an opaque, cryptographically random approval ID (for example, from a secure random token generator); do not use sequential/predictable IDs or expose the ID as a bearer credential through an extension/web API.
- User/principal and extension-session binding are not yet part of this grant schema. Add and validate that identity scope before exposing durable approvals through an authenticated extension transport.
- A grant is bound to request, task, step, tool ID and version, operation, idempotency key and a SHA-256 digest of the exact arguments.
- The store retains the digest and scope, not the raw argument object, cookies, credentials or page contents.
- Grants expire within 15 minutes of issuance. Expiration is checked both when issuing and consuming.
- `get(approval_id)` reloads an active, unexpired grant after application restart; consumed, revoked, expired or missing grants are not returned.
- `consume()` uses a SQLite `BEGIN IMMEDIATE` transaction, so two store instances/processes cannot both consume the same grant.
- `revoke()` atomically invalidates an unused grant. A consumed or revoked grant is terminal; scope cannot be edited and rows cannot be deleted through the database API/triggers.
- The approval store has its own metadata table rather than SQLite's database-wide `user_version`, so it can coexist in the same database file as the task ledger without sharing migration counters.
- File-backed databases use WAL, full synchronous commits, bounded busy timeout and best-effort owner-only database-file permissions. Protect the parent directory and backups as sensitive local user data.
- No grant-retention or deletion workflow is implemented. Define retention and secure disposal before production use.

Connect it to the existing dispatcher through the host-created registry:

```python
from core.agent import SQLiteApprovalStore, ToolRegistry

with SQLiteApprovalStore("data/automation/approvals.sqlite3") as approvals:
    registry = ToolRegistry(approval_consumer=approvals.consume)
    # Register adapters from trusted application code.
    # After the user confirms the exact operation, host code calls:
    # approvals.issue(approval_grant)
    # ToolDispatcher consumes the matching grant immediately before dispatch.
```

The store is opt-in and is not currently wired into `core.server` or a browser extension. A process crash after a grant is consumed but before an adapter effect is known still requires adapter idempotency/reconciliation. Durable permission policy, identity/session binding and a broader audit service remain separate integration work.

## AutomationRuntime (initial library integration)

`AutomationRuntime` is the first durable orchestration facade joining `SQLiteTaskStateStore`, `SQLiteApprovalStore`, `ToolRegistry` and `ToolDispatcher` in `core.agent.runtime`. It is a library component; it does not add an HTTP endpoint, browser-extension transport, autonomous planner or default server integration.

- Constructing the runtime opens both ledgers against the same SQLite file. The task store retains its own `user_version`; the approval store uses its separate metadata table.
- Adapters are registered only by trusted host code through `register_adapter()`. Plans and model output cannot register or directly execute a tool.
- Every runtime plan step must bind a `tool_id` and operation. The call must match the task request ID and that exact step binding.
- The host supplies the permission set on every `execute_step()` call. The runtime does not infer, persist or expand permissions.
- Missing approval for an approval-required operation moves the step to `WAITING_APPROVAL` and persists a cryptographically random opaque approval reference in the task snapshot; no adapter is called. The reference is a lookup handle, not execution permission.
- Trusted host code may call `issue_approval(call, user_confirmed=True)` only after explicit user consent. The runtime creates the durable grant for the reference already attached to the step, or rotates the reference if its earlier grant is expired, revoked or consumed. Grant lifetime is capped at 15 minutes.
- On execution, the runtime recovers the approval ID from the persisted task step when the caller omits it, reloads the active grant from SQLite and still sends it through `ToolDispatcher`. Dispatcher validation and atomic single-use consumption remain mandatory; `approve_step()` alone never authorizes an operation.
- Step start and final result are persisted. Successful tool output is returned to the caller but is not copied into task history. A failed step blocks its dependents; failure results preserve the adapter's side-effect uncertainty flag.
- `cancel_task()` records the cancellation request and signals active step events. The dispatcher checks cancellation before consuming approval and again before adapter entry; adapters must still cooperate with cancellation/deadlines.
- `recover_interrupted_tasks(exclusive_startup=True)` is explicit and never retries. Call it only when the host can guarantee that no other process/runtime is executing against the same database. Any persisted `RUNNING` step is marked failed with `side_effects_may_have_occurred=True`, because a crash cannot prove whether an external effect happened.
- `close()` refuses to close while synchronous adapter calls are active; signal/cancel and wait for those calls to finish before closing both stores.

Minimal shape:

```python
from core.agent import AutomationRuntime, PlanStep, ToolCall

with AutomationRuntime("data/automation/automation.sqlite3") as runtime:
    runtime.register_adapter(trusted_adapter)
    runtime.create_task(
        task_id="task-001",
        request_id="request-001",
        goal="Read the selected page.",
        steps=(
            PlanStep(
                "read", "Read the selected page.",
                tool_id="browser.reader", operation="read_page",
            ),
        ),
    )
    call = ToolCall(
        request_id="request-001",
        task_id="task-001",
        step_id="read",
        tool_id="browser.reader",
        tool_version="1.0.0",
        operation="read_page",
        arguments={"url": "https://example.test"},
    )
    result = runtime.execute_step(
        call, granted_permissions={"browser.read"}
    )
```

For approval-required operations, the first `execute_step()` returns `awaiting_approval=True` and an opaque `approval_id`; that reference is also persisted in the task snapshot. Only after the user confirms the exact operation may trusted host code call `issue_approval(call, user_confirmed=True)`. On restart, the host can call `execute_step()` without an approval ID; the runtime reloads the reference from the task and the active grant from SQLite.

This layer does not yet provide principal/user/session binding, approval UX, capability discovery by a model, multi-step planning, cross-process runtime ownership/leases, an application-wide audit service or an authenticated extension bridge. Do not expose it to extension/web callers until identity/session binding, origin checks, replay protection, request authentication, message limits and host permission policy are implemented. External side effects still require adapter-level idempotency and reconciliation.
