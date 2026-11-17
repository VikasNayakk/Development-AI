"""Pairing and request authentication primitives for the local BCE bridge.

This module deliberately does not create HTTP routes or relax the local server's
Host/Origin checks. A future server handler must call BridgeAuthenticator only
after validating the loopback Host and must never expose pairing codes except
through an owner-controlled local Studio surface.
"""
from __future__ import annotations

import hashlib
import hmac
import math
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping
from urllib.parse import urlsplit

from .contracts import RuntimeIdentity
from .sqlite_session_store import SQLiteSessionStore

_CHROME_EXTENSION_ID = re.compile(r"[a-p]{32}")
_PAIRING_CODE = re.compile(r"[A-Za-z0-9_-]{32,256}")
_BEARER_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,256}")
_SESSION_ID = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.:@/-]{0,127}")
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{16,128}")
_PAIRING_TTL_SECONDS = 300.0
_MAX_SESSION_TTL_SECONDS = 24 * 60 * 60


class BridgeAuthenticationError(ValueError):
    """A pairing or extension request failed authentication."""


@dataclass(frozen=True)
class PairedExtensionSession:
    """One-time pairing result; keep credential private to the extension host."""

    identity: RuntimeIdentity
    client_id: str
    expires_at_unix: float
    credential: str = field(repr=False)


class PairingManager:
    """Single-use owner-confirmed pairing codes for one local principal."""

    def __init__(
        self,
        sessions: SQLiteSessionStore,
        *,
        principal_id: str,
        clock: Callable[[], float] = time.time,
        session_ttl_seconds: float = 8 * 60 * 60,
    ) -> None:
        if not isinstance(sessions, SQLiteSessionStore):
            raise TypeError("sessions must be a SQLiteSessionStore.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if (
            isinstance(session_ttl_seconds, bool)
            or not isinstance(session_ttl_seconds, (int, float))
            or not math.isfinite(session_ttl_seconds)
            or session_ttl_seconds <= 0
            or session_ttl_seconds > _MAX_SESSION_TTL_SECONDS
        ):
            raise ValueError("session_ttl_seconds must be greater than 0 and at most 24 hours.")
        # RuntimeIdentity validates the principal identifier without creating a session.
        RuntimeIdentity(principal_id=principal_id, session_id="pairing-validation")
        self._sessions = sessions
        self._principal_id = principal_id
        self._clock = clock
        self._session_ttl_seconds = float(session_ttl_seconds)
        self._lock = threading.RLock()
        self._pairing_digest: str | None = None
        self._pairing_expires_at: float | None = None

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise BridgeAuthenticationError("Bridge clock is unavailable.")
        return float(value)

    def create_pairing_code(self) -> tuple[str, float]:
        """Create one five-minute code. A newer code invalidates the previous one."""
        with self._lock:
            now = self._now()
            code = secrets.token_urlsafe(32)
            self._pairing_digest = hashlib.sha256(code.encode("ascii")).hexdigest()
            self._pairing_expires_at = now + _PAIRING_TTL_SECONDS
            return code, self._pairing_expires_at

    def invalidate_pairing_code(self) -> None:
        with self._lock:
            self._pairing_digest = None
            self._pairing_expires_at = None

    def pair(
        self,
        pairing_code: str,
        *,
        client_id: str,
        owner_confirmed_client_id: str,
    ) -> PairedExtensionSession:
        """Consume the pairing code only after owner confirms the exact extension ID."""
        if not isinstance(pairing_code, str) or not _PAIRING_CODE.fullmatch(pairing_code):
            raise BridgeAuthenticationError("Pairing code is invalid or expired.")
        if not isinstance(client_id, str) or not _CHROME_EXTENSION_ID.fullmatch(client_id):
            raise BridgeAuthenticationError("The caller is not a valid Chrome extension origin.")
        if owner_confirmed_client_id != client_id:
            raise BridgeAuthenticationError("The owner did not confirm this exact extension ID.")

        supplied_digest = hashlib.sha256(pairing_code.encode("ascii")).hexdigest()
        with self._lock:
            now = self._now()
            if self._pairing_digest is None or self._pairing_expires_at is None:
                raise BridgeAuthenticationError("Pairing code is invalid, expired or already used.")
            if self._pairing_expires_at <= now:
                self._pairing_digest = None
                self._pairing_expires_at = None
                raise BridgeAuthenticationError("Pairing code is invalid, expired or already used.")
            if not hmac.compare_digest(self._pairing_digest, supplied_digest):
                raise BridgeAuthenticationError("Pairing code is invalid, expired or already used.")
            # Consume before writing the durable session. If storage fails, require a new code.
            self._pairing_digest = None
            self._pairing_expires_at = None

        identity = RuntimeIdentity(
            principal_id=self._principal_id,
            session_id=secrets.token_urlsafe(32),
        )
        credential = secrets.token_urlsafe(48)
        expires_at = now + self._session_ttl_seconds
        self._sessions.issue(
            identity,
            credential=credential,
            client_id=client_id,
            expires_at_unix=expires_at,
        )
        return PairedExtensionSession(
            identity=identity,
            client_id=client_id,
            expires_at_unix=expires_at,
            credential=credential,
        )


def _single_header(headers: Mapping[str, str], name: str) -> str | None:
    """Read exactly one header value; reject duplicates to avoid parser ambiguity."""
    if hasattr(headers, "get_all"):
        values = headers.get_all(name)
        if values is None or len(values) != 1:
            return None
        value = values[0]
        return value if isinstance(value, str) else None
    matches = [
        value for key, value in headers.items()
        if isinstance(key, str) and key.casefold() == name.casefold()
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        return None
    return matches[0]


def extension_id_from_origin(origin: str) -> str | None:
    """Accept only an exact chrome-extension origin with a canonical Chrome ID."""
    if not isinstance(origin, str) or len(origin) > 256:
        return None
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return None
    if (
        parsed.scheme != "chrome-extension"
        or not parsed.hostname
        or not _CHROME_EXTENSION_ID.fullmatch(parsed.hostname)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return parsed.hostname


class BridgeAuthenticator:
    """Authenticate extension requests and durably reject replayed request IDs."""

    def __init__(self, sessions: SQLiteSessionStore) -> None:
        if not isinstance(sessions, SQLiteSessionStore):
            raise TypeError("sessions must be a SQLiteSessionStore.")
        self._sessions = sessions

    def authenticate(self, headers: Mapping[str, str]) -> RuntimeIdentity | None:
        origin = _single_header(headers, "Origin")
        client_id = extension_id_from_origin(origin or "")
        authorization = _single_header(headers, "Authorization")
        session_id = _single_header(headers, "X-Aliens-Session")
        request_id = _single_header(headers, "X-Aliens-Request-Id")
        timestamp_text = _single_header(headers, "X-Aliens-Timestamp")
        if (
            client_id is None
            or authorization is None
            or session_id is None
            or request_id is None
            or timestamp_text is None
            or not _SESSION_ID.fullmatch(session_id)
            or not _REQUEST_ID.fullmatch(request_id)
        ):
            return None
        scheme, separator, credential = authorization.partition(" ")
        if (
            not separator
            or scheme.casefold() != "bearer"
            or not _BEARER_TOKEN.fullmatch(credential)
        ):
            return None
        try:
            timestamp = float(timestamp_text)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(timestamp):
            return None
        identity = self._sessions.authenticate(
            session_id,
            credential=credential,
            client_id=client_id,
        )
        if identity is None:
            return None
        if not self._sessions.consume_request_id(
            identity,
            client_id=client_id,
            request_id=request_id,
            timestamp_unix=timestamp,
        ):
            return None
        return identity
