"""Run the document ingestion worker for uploaded knowledge-base files."""

from __future__ import annotations

import argparse
import logging
import time

from kyuriagents.ingestion import KnowledgeBaseService
from kyuriagents.runtime import AgentRuntimeConfig

_LOGGER = logging.getLogger("ingestion_worker")


def main() -> None:
    """Run the ingestion worker."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Process queued knowledge-base ingestion jobs.")
    parser.add_argument("--once", action="store_true", help="Process at most one queued job and exit.")
    parser.add_argument("--idle-sleep", type=float, default=2.0, help="Seconds to sleep when no jobs are queued.")
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=None,
        help="Override DEEPAGENTS_INGESTION_JOB_TIMEOUT_SECONDS for stale running jobs.",
    )
    args = parser.parse_args()

    config = AgentRuntimeConfig.from_env()
    service = KnowledgeBaseService(config=config)
    wait_for_queue = config.enable_ingestion_redis_queue and not args.once
    while True:
        failed = service.fail_stale_jobs(max_age_seconds=args.job_timeout_seconds)
        if failed:
            _LOGGER.info("marked stale ingestion jobs failed: %s", failed)
        job = service.process_next_job(
            wait_for_queue=wait_for_queue,
            queue_timeout_seconds=config.ingestion_redis_block_timeout_seconds,
        )
        if job is None:
            if args.once:
                _LOGGER.info("no queued ingestion jobs")
                return
            time.sleep(float(args.idle_sleep))
            continue
        _LOGGER.info("processed ingestion job: %s", job.job_id)
        if args.once:
            return


if __name__ == "__main__":
    main()
