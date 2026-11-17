"""Automation agent contracts and orchestration foundations."""

from .bridge_auth import (
    BridgeAuthenticator,
    BridgeAuthenticationError,
    PairedExtensionSession,
    PairingManager,
    extension_id_from_origin,
)
from .contracts import (
    ApprovalGrant,
    RuntimeIdentity,
    ExecutionContext,
    ResultStatus,
    RiskLevel,
    SideEffect,
    ToolCall,
    ToolDispatcher,
    ToolManifest,
    ToolOperation,
    ToolRegistry,
    ToolResult,
    validate_json_value,
)
from .sqlite_approval_store import (
    ApprovalStoreError,
    ApprovalStoreIntegrityError,
    SQLiteApprovalStore,
)
from .sqlite_session_store import (
    SQLiteSessionStore,
    SessionStoreError,
    SessionStoreIntegrityError,
)
from .sqlite_task_state import (
    SQLiteTaskStateStore,
    TaskPersistenceError,
    TaskStoreIntegrityError,
)
from .task_state import (
    PlanStep,
    StaleTaskRevisionError,
    StepSnapshot,
    StepStatus,
    TaskEvent,
    TaskNotFoundError,
    TaskSnapshot,
    TaskStateError,
    TaskStateStore,
    TaskStatus,
)

from .runtime import (
    AutomationRuntime,
    RuntimeBusyError,
    RuntimeClosedError,
    RuntimeRecoveryError,
    RuntimeStepResult,
)

__all__ = [
    "ApprovalGrant", "RuntimeIdentity", "ExecutionContext",
    "BridgeAuthenticator", "BridgeAuthenticationError", "PairedExtensionSession",
    "PairingManager", "extension_id_from_origin", "ResultStatus", "RiskLevel",
    "SideEffect", "ToolCall", "ToolDispatcher", "ToolManifest",
    "ToolOperation", "ToolRegistry", "ToolResult", "validate_json_value",
    "PlanStep", "StaleTaskRevisionError", "StepSnapshot", "StepStatus",
    "TaskEvent", "TaskNotFoundError", "TaskSnapshot", "TaskStateError",
    "TaskStateStore", "TaskStatus", "SQLiteTaskStateStore",
    "TaskPersistenceError", "TaskStoreIntegrityError", "SQLiteApprovalStore",
    "ApprovalStoreError", "ApprovalStoreIntegrityError",
    "SQLiteSessionStore", "SessionStoreError", "SessionStoreIntegrityError",
    "AutomationRuntime", "RuntimeStepResult", "RuntimeBusyError",
    "RuntimeClosedError", "RuntimeRecoveryError",
]
