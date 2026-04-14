import argparse
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api.auth import api_key_store


@dataclass
class RequestResult:
    status: int
    error: Optional[str]
    headers: Dict[str, str]
    body: str
    duration_ms: float


def _utc_minute_window() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _perform_request(url: str, api_key: str, timeout_seconds: float) -> RequestResult:
    started_at = time.perf_counter()
    request = urllib_request.Request(
        url=url,
        method="GET",
        headers={"X-API-Key": api_key},
    )

    try:
        with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            duration_ms = (time.perf_counter() - started_at) * 1000.0
            return RequestResult(
                status=int(response.getcode()),
                error=None,
                headers=headers,
                body=body,
                duration_ms=duration_ms,
            )
    except urllib_error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        headers = {str(key).lower(): str(value) for key, value in exc.headers.items()}
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        return RequestResult(
            status=int(exc.code),
            error=None,
            headers=headers,
            body=body,
            duration_ms=duration_ms,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started_at) * 1000.0
        return RequestResult(
            status=0,
            error=str(exc),
            headers={},
            body="",
            duration_ms=duration_ms,
        )


def _read_int_header(headers: Dict[str, str], name: str) -> Optional[int]:
    raw_value = headers.get(name.lower())
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Concurrent rate limiter stress test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument(
        "--endpoint",
        default="/usage/limits",
        help="Protected endpoint to stress (must require API key)",
    )
    parser.add_argument("--requests", type=int, default=120, help="Total requests to send")
    parser.add_argument("--concurrency", type=int, default=40, help="Concurrent workers")
    parser.add_argument("--rpm-limit", type=int, default=50, help="Per-minute request limit for test tenant")
    parser.add_argument("--timeout-seconds", type=float, default=5.0, help="Per-request timeout")
    parser.add_argument(
        "--strict-single-minute",
        action="store_true",
        help="Fail if test execution crosses into the next minute",
    )
    parser.add_argument(
        "--keep-key",
        action="store_true",
        help="Keep generated API key active after run (default revokes key)",
    )
    parser.add_argument(
        "--verbose-errors",
        action="store_true",
        help="Print error response bodies for non-2xx/429 statuses",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    total_requests = max(1, int(args.requests))
    concurrency = max(1, int(args.concurrency))
    rpm_limit = max(1, int(args.rpm_limit))
    timeout_seconds = max(0.1, float(args.timeout_seconds))

    endpoint = "/" + str(args.endpoint).lstrip("/")
    base_url = str(args.base_url).rstrip("/")
    target_url = base_url + endpoint

    tenant_id = f"stress-{uuid.uuid4().hex[:12]}"
    key_name = f"stress-key-{tenant_id}"
    principal, raw_key = api_key_store.create_key(name=key_name, tenant_id=tenant_id)
    api_key_store.set_tenant_limits(
        tenant_id=tenant_id,
        requests_per_minute=rpm_limit,
        requests_per_day=max(total_requests * 5, rpm_limit * 10),
        max_jobs=5,
    )

    print(f"Target URL: {target_url}")
    print(f"Tenant ID: {tenant_id}")
    print(f"Requests: {total_requests}  Concurrency: {concurrency}  RPM limit: {rpm_limit}")

    started_window = _utc_minute_window()
    started_at = time.perf_counter()

    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_perform_request, target_url, raw_key, timeout_seconds) for _ in range(total_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
    ended_window = _utc_minute_window()

    success_count = sum(1 for item in results if 200 <= item.status < 300)
    too_many_count = sum(1 for item in results if item.status == 429)
    transport_error_count = sum(1 for item in results if item.status == 0)
    other_http_count = sum(1 for item in results if item.status not in {0, 429} and not (200 <= item.status < 300))

    minute_used_values = [
        value
        for value in (_read_int_header(item.headers, "x-ratelimit-used-minute") for item in results)
        if value is not None
    ]
    minute_remaining_values = [
        value
        for value in (_read_int_header(item.headers, "x-ratelimit-remaining-minute") for item in results)
        if value is not None
    ]

    max_minute_used = max(minute_used_values) if minute_used_values else None
    min_minute_remaining = min(minute_remaining_values) if minute_remaining_values else None

    print("--- Summary ---")
    print(f"Minute windows: {started_window} -> {ended_window}")
    print(f"Elapsed ms: {elapsed_ms:.2f}")
    print(f"2xx: {success_count}")
    print(f"429: {too_many_count}")
    print(f"Other HTTP: {other_http_count}")
    print(f"Transport errors: {transport_error_count}")
    if max_minute_used is not None:
        print(f"Max x-ratelimit-used-minute seen: {max_minute_used}")
    else:
        print("Max x-ratelimit-used-minute seen: <missing>")
    if min_minute_remaining is not None:
        print(f"Min x-ratelimit-remaining-minute seen: {min_minute_remaining}")
    else:
        print("Min x-ratelimit-remaining-minute seen: <missing>")

    passed = True
    failures: list[str] = []

    same_window = started_window == ended_window
    expected_success_same_window = min(total_requests, rpm_limit)

    if args.strict_single_minute and not same_window:
        passed = False
        failures.append("Execution crossed minute window while --strict-single-minute is enabled")

    if same_window:
        if success_count != expected_success_same_window:
            passed = False
            failures.append(
                f"Expected {expected_success_same_window} successful responses in one window, got {success_count}"
            )
        expected_429 = total_requests - expected_success_same_window
        if too_many_count != expected_429:
            passed = False
            failures.append(f"Expected {expected_429} 429 responses in one window, got {too_many_count}")

    if max_minute_used is not None and max_minute_used > rpm_limit:
        passed = False
        failures.append(
            f"Observed x-ratelimit-used-minute={max_minute_used}, which exceeds configured limit={rpm_limit}"
        )

    if success_count > 0 and not minute_used_values:
        passed = False
        failures.append("Rate-limit headers were not present on successful responses")

    if transport_error_count > 0:
        passed = False
        failures.append(f"Encountered {transport_error_count} transport errors")

    if other_http_count > 0:
        passed = False
        failures.append(f"Encountered {other_http_count} unexpected non-2xx/non-429 HTTP responses")

    if args.verbose_errors and (other_http_count > 0 or transport_error_count > 0):
        print("--- Error Details ---")
        for index, item in enumerate(results, start=1):
            if item.status in {0, 429} or (200 <= item.status < 300):
                continue
            preview = item.body[:240].replace("\n", " ")
            print(f"#{index} status={item.status} error={item.error or '<none>'} body={preview}")
        for index, item in enumerate(results, start=1):
            if item.status != 0:
                continue
            print(f"#{index} transport_error={item.error}")

    if not args.keep_key:
        api_key_store.revoke_key(principal.key_id)

    if passed:
        print("PASS: limiter behavior is consistent with configured caps for this run")
        return 0

    print("FAIL:")
    for failure in failures:
        print(f"- {failure}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
