import tempfile
import unittest
from pathlib import Path

from core.agent import (
    BridgeAuthenticator,
    BridgeAuthenticationError,
    PairingManager,
    RuntimeIdentity,
    SQLiteSessionStore,
    extension_id_from_origin,
)


CLIENT_ID = "a" * 32
OTHER_CLIENT_ID = "b" * 32


class BridgeAuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "bridge.sqlite3"
        self.now = [1000.0]
        self.sessions = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(self.sessions.close)
        self.pairing = PairingManager(
            self.sessions,
            principal_id="owner-opaque-id",
            clock=lambda: self.now[0],
            session_ttl_seconds=3600.0,
        )

    def pair(self):
        code, _ = self.pairing.create_pairing_code()
        return self.pairing.pair(
            code,
            client_id=CLIENT_ID,
            owner_confirmed_client_id=CLIENT_ID,
        )

    def headers(self, paired, *, request_id="request-id-0000001", timestamp=None, client_id=CLIENT_ID):
        return {
            "Origin": "chrome-extension://" + client_id,
            "Authorization": "Bearer " + paired.credential,
            "X-Aliens-Session": paired.identity.session_id,
            "X-Aliens-Request-Id": request_id,
            "X-Aliens-Timestamp": str(self.now[0] if timestamp is None else timestamp),
        }

    def test_pairing_code_is_single_use_and_binds_confirmed_extension(self):
        code, expiry = self.pairing.create_pairing_code()
        self.assertEqual(expiry, 1300.0)
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=OTHER_CLIENT_ID,
            )
        paired = self.pairing.pair(
            code,
            client_id=CLIENT_ID,
            owner_confirmed_client_id=CLIENT_ID,
        )
        self.assertEqual(paired.identity.principal_id, "owner-opaque-id")
        self.assertEqual(paired.client_id, CLIENT_ID)
        self.assertTrue(self.sessions.is_active(paired.identity))
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=CLIENT_ID,
            )
        self.assertNotIn(paired.credential, repr(paired))

    def test_new_pairing_code_invalidates_previous_code(self):
        old, _ = self.pairing.create_pairing_code()
        current, _ = self.pairing.create_pairing_code()
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                old,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=CLIENT_ID,
            )
        paired = self.pairing.pair(
            current,
            client_id=CLIENT_ID,
            owner_confirmed_client_id=CLIENT_ID,
        )
        self.assertTrue(self.sessions.is_active(paired.identity))

    def test_pairing_code_expires_and_can_be_invalidated(self):
        code, _ = self.pairing.create_pairing_code()
        self.now[0] += 300.0
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=CLIENT_ID,
            )
        self.now[0] = 1100.0
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=CLIENT_ID,
            )
        code, _ = self.pairing.create_pairing_code()
        self.pairing.invalidate_pairing_code()
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id=CLIENT_ID,
                owner_confirmed_client_id=CLIENT_ID,
            )

    def test_pairing_rejects_non_chrome_extension_id(self):
        code, _ = self.pairing.create_pairing_code()
        with self.assertRaises(BridgeAuthenticationError):
            self.pairing.pair(
                code,
                client_id="not-a-chrome-extension",
                owner_confirmed_client_id="not-a-chrome-extension",
            )

    def test_bridge_authenticator_accepts_valid_request_once(self):
        paired = self.pair()
        auth = BridgeAuthenticator(self.sessions)
        headers = self.headers(paired)
        self.assertEqual(auth.authenticate(headers), paired.identity)
        self.assertIsNone(auth.authenticate(headers))

    def test_bridge_authenticator_rejects_wrong_token_or_extension_origin(self):
        paired = self.pair()
        auth = BridgeAuthenticator(self.sessions)
        wrong_token = self.headers(paired, request_id="request-id-0000002")
        wrong_token["Authorization"] = "Bearer " + ("z" * 48)
        self.assertIsNone(auth.authenticate(wrong_token))

        wrong_origin = self.headers(paired, request_id="request-id-0000003", client_id=OTHER_CLIENT_ID)
        self.assertIsNone(auth.authenticate(wrong_origin))

    def test_bridge_authenticator_rejects_missing_duplicate_or_malformed_headers(self):
        paired = self.pair()
        auth = BridgeAuthenticator(self.sessions)
        missing = self.headers(paired)
        del missing["X-Aliens-Request-Id"]
        self.assertIsNone(auth.authenticate(missing))

        duplicate = self.headers(paired)
        duplicate["origin"] = duplicate["Origin"]
        self.assertIsNone(auth.authenticate(duplicate))

        malformed = self.headers(paired)
        malformed["X-Aliens-Timestamp"] = "nan"
        self.assertIsNone(auth.authenticate(malformed))

    def test_bridge_authenticator_rejects_stale_timestamp(self):
        paired = self.pair()
        auth = BridgeAuthenticator(self.sessions)
        stale = self.headers(paired, timestamp=900.0)
        self.assertIsNone(auth.authenticate(stale))

    def test_request_replay_protection_survives_store_restart(self):
        paired = self.pair()
        auth = BridgeAuthenticator(self.sessions)
        headers = self.headers(paired, request_id="durable-request-id-001")
        self.assertEqual(auth.authenticate(headers), paired.identity)
        self.sessions.close()

        reopened = SQLiteSessionStore(self.path, clock=lambda: self.now[0])
        self.addCleanup(reopened.close)
        reopened_auth = BridgeAuthenticator(reopened)
        self.assertIsNone(reopened_auth.authenticate(headers))

    def test_revoked_session_cannot_authenticate_or_submit_requests(self):
        paired = self.pair()
        self.assertTrue(self.sessions.revoke(paired.identity))
        auth = BridgeAuthenticator(self.sessions)
        self.assertIsNone(auth.authenticate(self.headers(paired)))

    def test_extension_origin_parser_is_strict(self):
        self.assertEqual(
            extension_id_from_origin("chrome-extension://" + CLIENT_ID),
            CLIENT_ID,
        )
        for origin in (
            "https://example.com",
            "chrome-extension://invalid",
            "chrome-extension://" + CLIENT_ID + "/path",
            "chrome-extension://" + CLIENT_ID + "?query=1",
            "chrome-extension://user@" + CLIENT_ID,
        ):
            self.assertIsNone(extension_id_from_origin(origin))


if __name__ == "__main__":
    unittest.main()
