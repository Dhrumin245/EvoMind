import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from cryptography.fernet import Fernet

from api.events import (
    EventManager,
    WEBHOOK_SECRET_ENCRYPTION_ENV_VAR,
    WEBHOOK_SECRET_ENCRYPTION_PREFIX,
)
from tests.tmp_utils import cleanup_path


class WebhookSecretStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = Path(f"tests/.tmp/webhook-secret-storage-{uuid.uuid4().hex}.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        cleanup_path(self.db_path)

    @staticmethod
    def _encryption_env(key: bytes) -> dict[str, str]:
        return {WEBHOOK_SECRET_ENCRYPTION_ENV_VAR: key.decode("utf-8")}

    def test_create_webhook_encrypts_secret_at_rest(self) -> None:
        key = Fernet.generate_key()
        with patch.dict(os.environ, self._encryption_env(key), clear=False):
            manager = EventManager(db_path=str(self.db_path))
            created = manager.create_webhook(
                tenant_id="tenant-1",
                url="https://93.184.216.34/incoming",
                secret="top-secret",
            )

            with manager._connect() as conn:
                row = conn.execute(
                    "SELECT secret FROM webhooks WHERE webhook_id = ?",
                    (created.webhook_id,),
                ).fetchone()

            stored_secret = row["secret"] if row is not None else None
            loaded = manager._get_webhook(created.webhook_id)

        self.assertIsNotNone(stored_secret)
        assert stored_secret is not None
        self.assertNotEqual(stored_secret, "top-secret")
        self.assertTrue(stored_secret.startswith(WEBHOOK_SECRET_ENCRYPTION_PREFIX))
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.secret, "top-secret")

    def test_manager_migrates_plaintext_webhook_secret_on_startup(self) -> None:
        key = Fernet.generate_key()
        with patch.dict(os.environ, self._encryption_env(key), clear=False):
            initial_manager = EventManager(db_path=str(self.db_path))
            with initial_manager._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO webhooks (
                        webhook_id,
                        tenant_id,
                        url,
                        description,
                        secret,
                        subscribed_events_json,
                        status,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, '[]', 'active', ?, ?)
                    """,
                    (
                        "webhook-legacy",
                        "tenant-1",
                        "https://hooks.example.com/legacy",
                        "legacy",
                        "plaintext-secret",
                        "2026-04-15T00:00:00Z",
                        "2026-04-15T00:00:00Z",
                    ),
                )
                conn.commit()

            migrated_manager = EventManager(db_path=str(self.db_path))
            loaded = migrated_manager._get_webhook("webhook-legacy")
            with migrated_manager._connect() as conn:
                row = conn.execute(
                    "SELECT secret FROM webhooks WHERE webhook_id = ?",
                    ("webhook-legacy",),
                ).fetchone()

        self.assertIsNotNone(row)
        assert row is not None
        self.assertTrue(str(row["secret"]).startswith(WEBHOOK_SECRET_ENCRYPTION_PREFIX))
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.secret, "plaintext-secret")

    def test_creating_secret_without_encryption_key_is_rejected(self) -> None:
        with patch.dict(os.environ, {WEBHOOK_SECRET_ENCRYPTION_ENV_VAR: ""}, clear=False):
            manager = EventManager(db_path=str(self.db_path))
            with self.assertRaisesRegex(RuntimeError, "Webhook secret encryption is not configured"):
                manager.create_webhook(
                    tenant_id="tenant-1",
                    url="https://93.184.216.34/incoming",
                    secret="top-secret",
                )

    def test_startup_fails_if_stored_secrets_exist_without_key(self) -> None:
        key = Fernet.generate_key()
        with patch.dict(os.environ, self._encryption_env(key), clear=False):
            manager = EventManager(db_path=str(self.db_path))
            manager.create_webhook(
                tenant_id="tenant-1",
                url="https://93.184.216.34/incoming",
                secret="top-secret",
            )

        with patch.dict(os.environ, {WEBHOOK_SECRET_ENCRYPTION_ENV_VAR: ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "Stored webhook secrets require encryption at rest"):
                EventManager(db_path=str(self.db_path))

    def test_file_based_encryption_key_is_supported(self) -> None:
        key = Fernet.generate_key()
        secret_file = self.db_path.with_suffix(".key")
        secret_file.write_text(key.decode("utf-8") + "\n", encoding="utf-8")

        with patch.dict(
            os.environ,
            {
                WEBHOOK_SECRET_ENCRYPTION_ENV_VAR: "",
                f"{WEBHOOK_SECRET_ENCRYPTION_ENV_VAR}_FILE": str(secret_file),
            },
            clear=False,
        ):
            manager = EventManager(db_path=str(self.db_path))
            created = manager.create_webhook(
                tenant_id="tenant-1",
                url="https://93.184.216.34/incoming",
                secret="from-file-secret",
            )

            loaded = manager._get_webhook(created.webhook_id)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.secret, "from-file-secret")


if __name__ == "__main__":
    unittest.main()
