import os
import sqlite3
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from api.auth import APIKeyStore
from api.events import EventManager
from api.job_manager import JobManager
from api.storage import (
    ManagedSqliteConnection,
    api_auth_db_path,
    api_events_db_path,
    api_jobs_db_path,
    data_dir,
    resolve_db_target,
    sqlite_connect,
    tenant_root_dir,
)
from tests.tmp_utils import cleanup_path


class StorageConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base_dir = Path(f"tests/.tmp/storage-config-{uuid.uuid4().hex}")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        cleanup_path(self.base_dir)

    def test_data_paths_follow_environment_overrides(self) -> None:
        custom_data = self.base_dir / "custom-data"
        custom_tenants = self.base_dir / "custom-tenants"
        custom_auth = self.base_dir / "db" / "auth.sqlite3"
        custom_events = self.base_dir / "db" / "events.sqlite3"
        custom_jobs = self.base_dir / "db" / "jobs.sqlite3"

        with patch.dict(
            os.environ,
            {
                "EVOMIND_DATA_DIR": str(custom_data),
                "EVOMIND_TENANT_ROOT_DIR": str(custom_tenants),
                "EVOMIND_API_AUTH_DB": str(custom_auth),
                "EVOMIND_API_EVENTS_DB": str(custom_events),
                "EVOMIND_API_JOBS_DB": str(custom_jobs),
            },
            clear=False,
        ):
            self.assertEqual(data_dir(), custom_data)
            self.assertEqual(tenant_root_dir(), custom_tenants)
            self.assertEqual(api_auth_db_path(), custom_auth)
            self.assertEqual(api_events_db_path(), custom_events)
            self.assertEqual(api_jobs_db_path(), custom_jobs)

    def test_sqlite_connect_applies_production_pragmas(self) -> None:
        db_path = self.base_dir / "runtime" / "app.db"

        conn = sqlite_connect(db_path, timeout=12.5)
        self.assertIsInstance(conn, ManagedSqliteConnection)
        try:
            journal_mode_row = conn.execute("PRAGMA journal_mode").fetchone()
            foreign_keys_row = conn.execute("PRAGMA foreign_keys").fetchone()
            busy_timeout_row = conn.execute("PRAGMA busy_timeout").fetchone()
            temp_store_row = conn.execute("PRAGMA temp_store").fetchone()
        finally:
            conn.close()

        assert journal_mode_row is not None
        assert foreign_keys_row is not None
        assert busy_timeout_row is not None
        assert temp_store_row is not None
        self.assertEqual(str(journal_mode_row[0]).lower(), "wal")
        self.assertEqual(int(foreign_keys_row[0]), 1)
        self.assertGreaterEqual(int(busy_timeout_row[0]), 12500)
        self.assertEqual(int(temp_store_row[0]), 2)

    def test_sqlite_context_manager_closes_connection(self) -> None:
        db_path = self.base_dir / "runtime" / "managed.db"

        with sqlite_connect(db_path) as conn:
            self.assertIsInstance(conn, ManagedSqliteConnection)
            conn.execute("SELECT 1").fetchone()

        with self.assertRaises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1").fetchone()

    def test_default_managers_use_storage_env_paths(self) -> None:
        custom_data = self.base_dir / "data-root"
        with patch.dict(os.environ, {"EVOMIND_DATA_DIR": str(custom_data)}, clear=False):
            auth_store = APIKeyStore()
            event_manager = EventManager()
            job_manager = JobManager(instance_id="storage-test")

        self.assertEqual(auth_store.db_path, custom_data / "api_auth.db")
        self.assertEqual(event_manager.db_path, custom_data / "api_events.db")
        self.assertEqual(job_manager.runtime_db_path, custom_data / "api_jobs.db")
        self.assertEqual(job_manager.root_dir, custom_data / "tenants")

    def test_resolve_db_target_prefers_control_plane_database_url(self) -> None:
        with patch.dict(
            os.environ,
            {"EVOMIND_CONTROL_PLANE_DB_URL": "postgresql://evomind:secret@example.com:5432/evomind"},
            clear=False,
        ):
            target = resolve_db_target(
                context="test",
                explicit_path=None,
                explicit_url=None,
                env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                default_path=self.base_dir / "fallback.sqlite3",
            )

        self.assertEqual(target.backend, "postgres")
        self.assertEqual(target.url, "postgresql://evomind:secret@example.com:5432/evomind")
        self.assertIsNone(target.path)

    def test_resolve_db_target_supports_file_based_database_url(self) -> None:
        secret_file = self.base_dir / "control-plane-db-url.txt"
        secret_file.write_text(
            "postgresql://evomind:secret@example.com:5432/evomind\n",
            encoding="utf-8",
        )

        with patch.dict(
            os.environ,
            {"EVOMIND_CONTROL_PLANE_DB_URL_FILE": str(secret_file)},
            clear=False,
        ):
            target = resolve_db_target(
                context="test",
                explicit_path=None,
                explicit_url=None,
                env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                default_path=self.base_dir / "fallback.sqlite3",
            )

        self.assertEqual(target.backend, "postgres")
        self.assertEqual(target.url, "postgresql://evomind:secret@example.com:5432/evomind")
        self.assertIsNone(target.path)

    def test_resolve_db_target_builds_database_url_from_components(self) -> None:
        password_file = self.base_dir / "control-plane-db-password.txt"
        password_file.write_text("secret-password\n", encoding="utf-8")

        with patch.dict(
            os.environ,
            {
                "EVOMIND_CONTROL_PLANE_DB_HOST": "postgres.internal",
                "EVOMIND_CONTROL_PLANE_DB_PORT": "5432",
                "EVOMIND_CONTROL_PLANE_DB_NAME": "evomind",
                "EVOMIND_CONTROL_PLANE_DB_USER": "service-account",
                "EVOMIND_CONTROL_PLANE_DB_PASSWORD_FILE": str(password_file),
                "EVOMIND_CONTROL_PLANE_DB_SSLMODE": "require",
            },
            clear=False,
        ):
            target = resolve_db_target(
                context="test",
                explicit_path=None,
                explicit_url=None,
                env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                default_path=self.base_dir / "fallback.sqlite3",
            )

        self.assertEqual(target.backend, "postgres")
        self.assertEqual(
            target.url,
            "postgresql://service-account:secret-password@postgres.internal:5432/evomind?sslmode=require",
        )
        self.assertIsNone(target.path)

    def test_production_requires_postgres_unless_explicitly_overridden(self) -> None:
        with patch.dict(
            os.environ,
            {"EVOMIND_ENV": "production"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "must use PostgreSQL in production"):
                resolve_db_target(
                    context="test",
                    explicit_path=self.base_dir / "runtime.sqlite3",
                    explicit_url=None,
                    env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                    default_path=self.base_dir / "fallback.sqlite3",
                )

        with patch.dict(
            os.environ,
            {
                "EVOMIND_ENV": "production",
                "EVOMIND_ALLOW_SQLITE_IN_PRODUCTION": "true",
            },
            clear=False,
        ):
            target = resolve_db_target(
                context="test",
                explicit_path=self.base_dir / "runtime.sqlite3",
                explicit_url=None,
                env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                default_path=self.base_dir / "fallback.sqlite3",
            )

        self.assertEqual(target.backend, "sqlite")
        self.assertEqual(target.path, self.base_dir / "runtime.sqlite3")


if __name__ == "__main__":
    unittest.main()
