import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from api import server
from api.auth import API_KEY_ROLE_OPERATOR, API_KEY_SCOPE_JOBS_READ, APIKeyStore
from api.schemas import ApiKeyCreateRequest
from tests.postgres_utils import postgres_url, reset_tables


class APIKeyHttpRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_url = postgres_url()
        reset_tables(self.db_url)
        self.store = APIKeyStore(db_url=self.db_url)
        self.admin_principal, self.admin_key = self.store.create_key(
            name="admin",
            tenant_id="tenant-http",
        )
        self.server_patch = patch.object(server, "api_key_store", self.store)
        self.server_patch.start()

    def tearDown(self) -> None:
        self.server_patch.stop()
        reset_tables(self.db_url)

    def test_create_list_and_revoke_api_key_for_current_tenant(self) -> None:
        create_payload = asyncio.run(
            server.create_api_key(
                ApiKeyCreateRequest(name="dashboard", role="reader", scopes=[API_KEY_SCOPE_JOBS_READ]),
                principal=self.admin_principal,
            )
        )

        self.assertTrue(create_payload.api_key.startswith("evm_"))
        created_key_id = create_payload.key.key_id
        self.assertEqual(create_payload.key.tenant_id, "tenant-http")
        self.assertEqual(create_payload.key.role, "reader")

        list_payload = asyncio.run(server.list_api_keys(principal=self.admin_principal))
        listed_ids = {item.key_id for item in list_payload.items}
        self.assertIn(self.admin_principal.key_id, listed_ids)
        self.assertIn(created_key_id, listed_ids)

        delete_payload = asyncio.run(server.delete_api_key(created_key_id, principal=self.admin_principal))
        self.assertEqual(delete_payload.key_id, created_key_id)
        self.assertTrue(delete_payload.deleted)
        self.assertEqual(self.store.get_key(created_key_id).status, "revoked")  # type: ignore[union-attr]

    def test_non_admin_scoped_key_cannot_manage_api_keys(self) -> None:
        operator, _ = self.store.create_key(
            name="operator",
            tenant_id="tenant-http",
            role=API_KEY_ROLE_OPERATOR,
            scopes=[API_KEY_SCOPE_JOBS_READ],
        )

        with self.assertRaises(HTTPException) as exc_info:
            self.store.require_permission(operator, "GET", "/auth/keys")
        self.assertEqual(exc_info.exception.status_code, 403)
        self.assertIn("auth:read", str(exc_info.exception.detail))


if __name__ == "__main__":
    unittest.main()
