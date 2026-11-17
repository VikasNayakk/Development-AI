# Extension Agent — AI Workstream Backlog

**Owner plan captured:** 2026-11-17  
**Scope:** Development-AI responsibilities required by the BCE / Browser Control Extension product.  
**Status:** Planning baseline derived from the owner's current partial product plan. Future requirements may add, remove or reprioritize work.

This backlog separates the proprietary AI's responsibilities from browser execution. BCE remains responsible for reading browser/application state and performing approved extension actions. Development-AI is responsible for intent understanding, capability discovery, task classification, planning, decision support, workflow selection, safe recovery and explanations. Neither an LLM response nor a browser page may grant itself permissions.

## Product behavior to deliver

A user or authorized integration gives the system a goal. The AI determines whether the task is:
1. a small, supported direct action that BCE can safely perform;
2. a task already covered by an approved, versioned workflow;
3. a task that needs clarification, new workflow authoring or a human decision; or
4. unsupported / unsafe, in which case the system explains why and does not improvise an action.

For Teams intake, the AI monitors only the groups and task scope explicitly selected by the owner. When a likely task for the owner appears, the AI presents the source group, message, task interpretation, suggested workflow, target and risk. It waits for a scoped approval before execution. When no workflow exists, it proposes a workflow-authoring interview; it does not silently create or execute a workflow.

When a new workflow is needed, the AI gathers the process from the owner and produces a structured, versioned workflow specification. A ChatGPT-assisted implementation may be requested through BCE, but generated code and downloaded workflow packages are untrusted until they pass validation, tests, review and explicit activation approval.

## Architecture boundary

- **Model:** interprets Hindi, English and Hinglish requests; classifies intent; proposes plans; explains uncertainty. Model output is a proposal, never an executable authorization.
- **Policy / approval:** deterministic code verifies principal, session, permissions, resource scope, operation risk, approval expiry and task state.
- **Agent runtime:** uses the existing AutomationRuntime, durable task/approval stores and ToolDispatcher; owns task lifecycle, bounded scheduling, cancellation, retries and event history.
- **Tool registry:** exposes only host-registered, versioned manifests and real capability/health state.
- **Adapters:** BCE and future workflow/application adapters execute narrowly scoped operations, validate locally and return structured evidence.
- **Workflow repository:** stores versioned workflow specifications and validated packages. New code is quarantined until reviewed and activated.
- **Transport:** extension-to-AI communication must be authenticated, loopback/local-only unless a future threat model explicitly approves otherwise. Do not expose the Studio server to a LAN as a shortcut.

## Existing foundation already present

The following are foundations, not proof that the full product is complete:
- Versioned tool contracts, schema validation, registry and dispatcher exist.
- Process-local task state, durable SQLite task ledger and durable exact-scope single-use approval grants exist.
- AutomationRuntime was merged to main in commit 2d4c9131239d44f0b8514c93a87ef1fa5abb6a07. It composes the durable stores and dispatcher as an opt-in library component.
- Exact PR #24 head passed 209 Python tests; post-merge main CI passed at the merge commit.
- The runtime is **not** wired into core.server or BCE. Principal/session binding, authenticated bridge, autonomous planning, user approval UI, app adapters and workflow runtime remain pending.
- The current from-scratch research model is not yet a dependable general assistant or task classifier. Do not route real work through it until it passes the task-understanding and tool-use evaluation gates.

## Delivery backlog

### AI-1 — Identity, trust and authorization scope

**Implementation status:** backend identity binding and durable session expiry/revocation are implemented on feature branch `feature/principal-session-binding`; authenticated extension identity sourcing and approval confirmation UX remain pending CI/review.

- [ ] Define a stable owner/principal identity source and its trust boundary.
- [ ] Define extension installation identity and per-browser session identity; do not treat a tab ID or conversation ID as a user identity.
- [x] Bind task records and approval grants to principal + extension session. Workflow activations and audit events are not implemented yet.
- [x] Extend approval scope to bind principal/session, exact arguments digest, tool/version/operation, task/step, request and idempotency key. Target-resource display/UX remains pending.
- [x] Add durable session expiry and revocation; revoked sessions invalidate pending approvals. Host sign-out wiring remains pending.
- [ ] Make approval confirmation display the exact operation, target, side effects and limits. No blanket approval for unrelated future work.
- [x] Add tests for cross-user/session mismatch, session expiry/revocation, approval replay and changed arguments. Authenticated bridge replay tests remain part of AI-2.
- **Acceptance:** a grant created by one user/session cannot authorize another user's task or another extension session, including after restart. Invalid identity or scope fails closed.

### AI-2 — Authenticated local bridge to BCE

- [ ] Select and document a Chrome-supported transport (for example, extension-native messaging to a loopback host) and a threat model before exposing an endpoint.
- [ ] Keep all local service bindings on loopback; reject non-loopback hosts and unexpected Host / Origin values.
- [ ] Add a pairing handshake with a high-entropy, revocable session credential; never place long-lived secrets in page content, model prompts or URLs.
- [ ] Authenticate every request and bind it to the paired extension identity and active principal.
- [ ] Add request IDs, nonces / replay protection, strict timestamps, bounded payload sizes, request deadlines and rate limits.
- [ ] Define versioned request / response schemas for status, capabilities, task proposal, approval request, dispatch, cancellation and result reporting.
- [ ] Reject unknown message types, extra fields where prohibited, malformed JSON, oversized arrays and unsupported schema versions.
- [ ] Add disconnect, extension reload, server restart, permission removal and cancellation behavior.
- [ ] Add integration tests for hostile origins, replayed messages, expired credentials, malformed payloads, timeouts and server restarts.
- **Acceptance:** only a paired extension session can send allowed messages; an unpaired webpage, stale session or replayed request cannot invoke a tool. The public/LAN network cannot reach this bridge.

### AI-3 — Capability discovery and deterministic task classification

- [ ] Load tool manifests only from trusted host-registered adapters; include version, operation, input/output schema, required permissions, risk, side effects, deadline, cancellation, idempotency and availability.
- [ ] Expose a read-only capability discovery API to the model; the model cannot register tools or edit their permissions.
- [ ] Define task classes: direct native action, existing workflow, workflow needed, clarification required, human-only / unsupported, unsafe.
- [ ] Classify the goal against real manifest and workflow registry capabilities, not names guessed from model memory.
- [ ] Return a structured decision: class, evidence, proposed tool/workflow, required inputs, assumptions, confidence and reason for escalation.
- [ ] Ask one targeted clarification when target, account, file, date, record or intended side effect is ambiguous.
- [ ] Support Hindi, English and Hinglish inputs without changing the user's target or numerical values.
- [ ] Keep message/page/document contents marked as untrusted data; prompt-injection text cannot alter policy, register tools or issue approvals.
- **Acceptance:** for a fixed benchmark, classification is deterministic at the policy boundary; unknown tools are never selected for dispatch, and low-confidence/ambiguous results ask the user.

### AI-4 — Plan generation, validation and execution supervision

- [ ] Define a plan schema with task/step IDs, dependencies, tool/version/operation, validated arguments, target resource, preconditions, postconditions, risk, side effects, timeout and idempotency.
- [ ] Make the model produce a plan proposal; validate structure and dependencies with deterministic code.
- [ ] Check the plan against live tool manifests, user permissions, session scope and approved workflow versions.
- [ ] Separate read-only observation from write / submit / delete / external communication operations.
- [ ] Require human confirmation for externally visible, destructive, financial, security-sensitive or hard-to-reverse actions.
- [ ] Execute eligible steps through AutomationRuntime / ToolDispatcher only; never run model-generated shell, JavaScript or arbitrary Python as an action.
- [ ] Apply bounded concurrency, deadlines, cancellation and per-resource locks; preserve task progress across restarts.
- [ ] Verify each postcondition using adapter evidence. A successful click or adapter return alone is not sufficient proof.
- [ ] Retry only when the operation and adapter contract explicitly establish safe idempotency; otherwise mark uncertainty and ask.
- [ ] Provide a readable task timeline: proposed, approved, started, verified, failed, blocked, cancelled or needs review.
- **Acceptance:** test plans cannot bypass schema/permission/approval checks; task state and event history remain consistent under cancellation, restart, partial failure and stale revisions.

### AI-5 — Teams task intake, ownership matching and approval inbox

- [ ] Define the BCE event contract for selected Teams groups, message/thread identity, sender, timestamp and permitted message context.
- [ ] Respect explicit group selection and minimum required permissions; never scan all groups by default.
- [ ] Detect messages that plausibly assign a task to the owner. Distinguish direct mention, explicit assignment, named reference and ambiguous name collision.
- [ ] Extract task description, requester, group, due date, priority, dependencies, linked resources and uncertainty; retain source message references.
- [ ] Deduplicate by stable message/thread identity and update an existing task when a source message is edited.
- [ ] Detect prompt-injection and untrusted instructions in Teams messages; message text cannot grant permission or override the owner.
- [ ] Match the task against current capabilities and approved workflows; show why each match is proposed.
- [ ] Send the owner an approval proposal with source group/message, normalized task, suggested workflow, target resource, risk and expected side effects.
- [ ] Support approve, reject, defer, ask for clarification and "no matching workflow".
- [ ] Preserve approval identity/session binding and audit the decision; approval must be scoped to the exact task and plan version.
- **Acceptance:** a matching Teams task creates a proposal only; no workflow executes before the owner approves the exact operation. Duplicate messages do not create duplicate execution.

### AI-6 — Workflow specification, authoring interview and registry

- [ ] Define a declarative workflow schema: stable ID, name, version, purpose, triggers, required inputs, preconditions, steps, dependencies, tool operations, output schema and postconditions.
- [ ] Define explicit failure, retry, compensation / rollback and manual-review policies for each step.
- [ ] Create an owner interview that asks for missing business rules, field mappings, examples, edge cases and expected output.
- [ ] Convert answers into a workflow specification and show it to the owner for correction before implementation starts.
- [ ] Validate each step against registered capabilities; reject arbitrary code and unknown tool operations.
- [ ] Store workflow specs, test fixtures, package versions and activation state in a versioned registry.
- [ ] Separate draft, review, approved, active, deprecated and revoked states.
- [ ] Require an explicit owner decision before activating a new workflow or replacing an active version.
- [ ] Support rollback to a prior approved version and preserve the exact version used by every task.
- **Acceptance:** every active workflow is versioned, schema-valid, capability-bound, testable and explicitly approved. Draft workflows cannot be triggered by production tasks.

### AI-7 — ChatGPT-assisted workflow implementation and GitHub intake

- [ ] Generate a comprehensive implementation brief from the approved workflow specification, not directly from an unverified Teams message.
- [ ] Ask BCE to submit the brief to the user-selected ChatGPT conversation; wait for the conversation's response to finish before asking follow-up prompts.
- [ ] Require generated implementation to be delivered as a versioned GitHub branch / pull request, not silently committed into the active workflow registry.
- [ ] Fetch only the expected repository, branch and PR; verify repository identity, base branch, changed-file allowlist and commit SHA.
- [ ] Treat the downloaded code and workflow package as untrusted input. Do not execute code during retrieval.
- [ ] Run static checks, workflow-schema validation, dependency/license review and tests in a restricted CI/sandbox with no user secrets.
- [ ] Use fake adapters and fixture data; prohibit production credentials, real external side effects and unrestricted network access in author tests.
- [ ] Produce a review report with changed files, permissions, effects, failed tests and unresolved risks.
- [ ] Require owner review/approval before merging or activating; if the user explicitly delegates merge, still require all deterministic gates and use an exact expected head SHA.
- [ ] Pin the activated commit/package digest; support revocation and rollback.
- **Acceptance:** a generated workflow cannot become active merely because ChatGPT says "done" or GitHub contains a commit. Exact PR/commit, static validation, test pass and owner approval are mandatory.

### AI-8 — Result verification, explanation, memory and continuous improvement

- [ ] Define structured adapter evidence for each supported operation and its target resource.
- [ ] Compare observed postconditions with the requested outcome and report partial completion honestly.
- [ ] Distinguish verified success, unverified, partial, failed, denied, timed out, cancelled and uncertain external side effect.
- [ ] On failure, explain the safe recovery choices; do not automatically retry a non-idempotent action.
- [ ] Store only the minimum task context needed; keep request context, task state, user-approved memory and tool secrets separate.
- [ ] Add owner-approved reusable process notes and workflow preferences; no silent learning from private Teams messages or work history.
- [ ] Add retention, export and deletion rules for task/approval/audit information.
- [ ] Provide feedback controls to mark task classification, workflow match and result report as correct/incorrect.
- **Acceptance:** each report cites evidence from the adapter or states that the result is not independently verified. No private task content becomes training data without a separate explicit governance decision.

### AI-9 — Model capability, instruction tuning and evaluation

- [ ] Preserve the from-scratch model requirement; do not substitute a pretrained checkpoint or hosted inference service.
- [ ] Build a rights-cleared, provenance-tracked instruction corpus for task classification, plan schemas, tool manifests, approval explanations, recovery and bilingual interaction.
- [ ] Create a held-out evaluation set for Hindi, English and Hinglish tasks; keep test items out of training.
- [ ] Benchmark intent classification, capability selection, plan schema validity, required clarification, permission-boundary compliance and recovery recommendations.
- [ ] Evaluate against prompt injection, ambiguous targets, malicious tool/page text, wrong user identity, stale approvals and duplicate dispatch cases.
- [ ] Add supervised fine-tuning / instruction-tuning only after the data and baseline evaluation are ready.
- [ ] Measure model quality separately from deterministic policy correctness; the policy layer remains authoritative even if the model score improves.
- [ ] Define release thresholds and regression gates before the model can propose real tasks.
- **Acceptance:** model output meets documented held-out thresholds and never controls permission decisions. Until then, real actions remain host-approved deterministic workflows.

### AI-10 — Release, observability and operational readiness

- [ ] Add structured redacted audit events for request, classification, plan, approval, dispatch, result, workflow version and recovery.
- [ ] Define health checks and compatibility matrix for AI runtime, BCE protocol and workflow adapter versions.
- [ ] Add rate limits, storage quotas, event retention and recovery drills.
- [ ] Add operator diagnostics for missing permissions, disconnected bridge, stale tabs, unavailable tools and version mismatch.
- [ ] Test full user journeys using fake Teams and fake application adapters before enabling real write adapters.
- [ ] Document installation, pairing, permissions, backup, recovery, stop-all and incident response.
- [ ] Run a security review and owner acceptance test before enabling a production adapter.
- **Acceptance:** release has reproducible CI, a threat model, recovery/rollback procedure, auditable approvals and an accessible emergency stop.

## Priority and dependency order

1. **Do not expose AutomationRuntime yet.** First complete AI-1 (identity/session scope) and AI-2 (authenticated bridge). These are hard security gates.
2. Build AI-3 and AI-4 around the existing runtime and tool manifests using fake adapters.
3. Add AI-5 Teams intake/approval inbox as a proposal-only flow.
4. Build AI-6 workflow specification/registry; only after that add AI-7 ChatGPT/GitHub implementation intake.
5. Develop AI-8 verification/memory/audit alongside execution, not after a production release.
6. AI-9 model work proceeds against the evaluation suite; model capability must not be assumed from the fact that the runtime works.
7. AI-10 production readiness gates every real application adapter.
8. Only after BCE + AI pass these gates should the user-specific Excel, To Do, OneDrive, OneNote and other business workflows be developed and activated.

## Current AI project meter

The wider Development-AI project has a separate nine-milestone engineering roadmap. At this baseline, **3 of 9 major milestones are fully complete (33.3%)**; several other foundations are partially complete. The merge of AutomationRuntime improves the automation foundation but does not by itself complete a major milestone. The extension-agent backlog above is a scoped workstream and is not rolled into the model's 33.3% meter.

Do not claim the project is a capable assistant or ready for live office automation until the model evaluation, identity/session binding, authenticated bridge, approval flow, workflow registry, adapters and end-to-end tests are complete.

## Change management

This is a first version based on the owner's partial plan as of 2026-11-17. When the owner adds requirements, record them here and in BCE's Extensions/BCE/PROJECT-ROADMAP.md, identify cross-repository dependencies, and update task counts. A task is complete only after implementation and its acceptance check pass; documentation alone does not complete an implementation task.
