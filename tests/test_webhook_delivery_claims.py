import unittest

from api.events import EventManager
from tests.postgres_utils import postgres_url, reset_tables


class WebhookDeliveryClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db_url = postgres_url()
        reset_tables(self.db_url)
        self.manager = EventManager(db_url=self.db_url)

    def tearDown(self) -> None:
        reset_tables(self.db_url)

    def _insert_delivery(
        self,
        delivery_id: str,
        status: str,
        next_retry_at: str | None,
        processing_started_at: str | None = None,
        claim_token: str | None = None,
    ) -> None:
        with self.manager._connect() as conn:
            conn.execute(
                """
                INSERT INTO webhook_deliveries (
                    delivery_id,
                    webhook_id,
                    event_id,
                    tenant_id,
                    job_id,
                    event_type,
                    status,
                    attempt_count,
                    max_attempts,
                    next_retry_at,
                    delivered_at,
                    last_error,
                    processing_started_at,
                    claim_token,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 0, 5, ?, NULL, NULL, ?, ?, ?, ?)
                """,
                (
                    delivery_id,
                    "webhook-1",
                    "event-1",
                    "tenant-1",
                    "job-1",
                    "job.started",
                    status,
                    next_retry_at,
                    processing_started_at,
                    claim_token,
                    "2026-04-12T10:00:00Z",
                    "2026-04-12T10:00:00Z",
                ),
            )
            conn.commit()

    def _read_delivery_claim_state(self, delivery_id: str) -> tuple[str, str | None, str | None]:
        with self.manager._connect() as conn:
            row = conn.execute(
                """
                SELECT status, processing_started_at, claim_token
                FROM webhook_deliveries
                WHERE delivery_id = ?
                """,
                (delivery_id,),
            ).fetchone()
        assert row is not None
        return str(row["status"]), row["processing_started_at"], row["claim_token"]

    def test_due_delivery_is_claimed_only_once(self) -> None:
        due_at = "2026-04-12T12:00:00Z"
        self._insert_delivery(
            delivery_id="delivery-1",
            status="pending",
            next_retry_at=due_at,
        )

        first_claim = self.manager._claim_due_deliveries(due_at=due_at, limit=10)
        second_claim = self.manager._claim_due_deliveries(due_at=due_at, limit=10)

        self.assertEqual(len(first_claim), 1)
        self.assertEqual(first_claim[0].delivery_id, "delivery-1")
        self.assertEqual(second_claim, [])

        status, processing_started_at, claim_token = self._read_delivery_claim_state("delivery-1")
        self.assertEqual(status, "processing")
        self.assertEqual(processing_started_at, due_at)
        self.assertTrue(claim_token)

    def test_stale_processing_delivery_is_reclaimed(self) -> None:
        due_at = "2026-04-12T12:00:00Z"
        self._insert_delivery(
            delivery_id="delivery-stale",
            status="processing",
            next_retry_at=None,
            processing_started_at="2026-04-12T11:40:00Z",
            claim_token="old-claim",
        )

        claimed = self.manager._claim_due_deliveries(
            due_at=due_at,
            limit=10,
            processing_lease_seconds=300,
        )

        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0].delivery_id, "delivery-stale")

        status, processing_started_at, claim_token = self._read_delivery_claim_state("delivery-stale")
        self.assertEqual(status, "processing")
        self.assertEqual(processing_started_at, due_at)
        self.assertNotEqual(claim_token, "old-claim")

    def test_setting_delivery_state_clears_claim_metadata(self) -> None:
        due_at = "2026-04-12T12:00:00Z"
        self._insert_delivery(
            delivery_id="delivery-cleanup",
            status="pending",
            next_retry_at=due_at,
        )

        claimed = self.manager._claim_due_deliveries(due_at=due_at, limit=10)
        self.assertEqual(len(claimed), 1)

        self.manager._set_delivery_state(
            delivery_id="delivery-cleanup",
            status_value="delivered",
            attempt_count=1,
            next_retry_at=None,
            delivered_at=due_at,
            last_error=None,
        )

        status, processing_started_at, claim_token = self._read_delivery_claim_state("delivery-cleanup")
        self.assertEqual(status, "delivered")
        self.assertIsNone(processing_started_at)
        self.assertIsNone(claim_token)


if __name__ == "__main__":
    unittest.main()
