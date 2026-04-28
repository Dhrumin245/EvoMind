from api.job_manager import JobManager


def main() -> int:
    try:
        manager = JobManager(instance_id="worker-healthcheck")
        return 0 if manager.has_active_worker("training") else 1
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
