import os
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from api.storage import data_dir, resolve_db_target, tenant_root_dir
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

        with patch.dict(
            os.environ,
            {
                "EVOMIND_DATA_DIR": str(custom_data),
                "EVOMIND_TENANT_ROOT_DIR": str(custom_tenants),
            },
            clear=False,
        ):
            self.assertEqual(data_dir(), custom_data)
            self.assertEqual(tenant_root_dir(), custom_tenants)

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
                default_path=None,
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
                default_path=None,
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
                default_path=None,
            )

        self.assertEqual(target.backend, "postgres")
        self.assertEqual(
            target.url,
            "postgresql://service-account:secret-password@postgres.internal:5432/evomind?sslmode=require",
        )
        self.assertIsNone(target.path)

    def test_filesystem_database_paths_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no longer supports filesystem database paths"):
            resolve_db_target(
                context="test",
                explicit_path=self.base_dir / "runtime.pg",
                explicit_url=None,
                env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                default_path=None,
            )

    def test_missing_database_url_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires PostgreSQL"):
                resolve_db_target(
                    context="test",
                    explicit_path=None,
                    explicit_url=None,
                    env_url_names=("EVOMIND_API_AUTH_DB_URL",),
                    default_path=None,
                )


if __name__ == "__main__":
    unittest.main()
