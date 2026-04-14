import ipaddress
import unittest
from unittest.mock import patch

from api.events import EventManager


class WebhookUrlValidationTests(unittest.TestCase):
    def test_public_ip_literal_is_allowed(self) -> None:
        url = "https://93.184.216.34/webhooks/evomind"

        normalized = EventManager._validate_webhook_url(url)

        self.assertEqual(normalized, url)

    def test_private_ip_literal_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "publicly routable IP addresses",
        ):
            EventManager._validate_webhook_url("http://127.0.0.1:8000/hook")

    def test_localhost_hostname_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "publicly reachable",
        ):
            EventManager._validate_webhook_url("https://localhost/webhook")

    @patch.object(EventManager, "_resolve_webhook_host_addresses")
    def test_hostname_resolving_to_public_ip_is_allowed(self, mock_resolve) -> None:
        mock_resolve.return_value = [ipaddress.ip_address("93.184.216.34")]

        normalized = EventManager._validate_webhook_url("https://hooks.example.com/path")

        self.assertEqual(normalized, "https://hooks.example.com/path")
        mock_resolve.assert_called_once_with("hooks.example.com")

    @patch.object(EventManager, "_resolve_webhook_host_addresses")
    def test_hostname_resolving_to_private_ip_is_rejected(self, mock_resolve) -> None:
        mock_resolve.return_value = [ipaddress.ip_address("10.0.0.5")]

        with self.assertRaisesRegex(
            ValueError,
            "publicly routable IP addresses",
        ):
            EventManager._validate_webhook_url("https://internal.example.com/hook")

    @patch.object(EventManager, "_resolve_webhook_host_addresses")
    def test_unresolvable_hostname_is_rejected(self, mock_resolve) -> None:
        mock_resolve.side_effect = ValueError("Webhook host could not be resolved")

        with self.assertRaisesRegex(
            ValueError,
            "could not be resolved",
        ):
            EventManager._validate_webhook_url("https://missing.example.com/hook")


if __name__ == "__main__":
    unittest.main()
