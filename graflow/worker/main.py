#!/usr/bin/env python3
"""TaskWorker independent process entry point."""

import argparse
import logging
import os
import signal
import sys
import time
from typing import Any, Dict

from graflow.queue.distributed import DistributedTaskQueue

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def create_redis_queue(redis_config: Dict[str, Any], logger: logging.Logger) -> DistributedTaskQueue:
    """RedisTaskQueue factory from configuration."""
    try:
        import redis

        # Create Redis client
        redis_client = redis.Redis(
            host=redis_config["host"], port=redis_config["port"], db=redis_config.get("db", 0), decode_responses=True
        )

        # Test connection
        redis_client.ping()
        logger.info(f"Connected to Redis at {redis_config['host']}:{redis_config['port']}")

        # RedisTaskQueue no longer needs execution_context
        # Tasks are retrieved from Graph via GraphStore
        return DistributedTaskQueue(redis_client=redis_client, key_prefix=redis_config.get("key_prefix", "graflow"))

    except ImportError:
        logger.error("redis package is required for Redis TaskQueue")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TaskWorker independent process", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Worker configuration
    parser.add_argument(
        "--worker-id", default=os.environ.get("WORKER_ID", f"worker_{os.getpid()}"), help="Unique worker identifier"
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=int(os.environ.get("MAX_CONCURRENT_TASKS", "4")),
        help="Maximum number of concurrent tasks",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.environ.get("POLL_INTERVAL", "0.1")),
        help="Polling interval in seconds",
    )

    # Redis configuration
    parser.add_argument("--redis-host", default=os.environ.get("REDIS_HOST", "localhost"), help="Redis host")
    parser.add_argument("--redis-port", type=int, default=int(os.environ.get("REDIS_PORT", "6379")), help="Redis port")
    parser.add_argument(
        "--redis-db", type=int, default=int(os.environ.get("REDIS_DB", "0")), help="Redis database number"
    )
    parser.add_argument(
        "--redis-key-prefix",
        default=os.environ.get("REDIS_KEY_PREFIX", "graflow"),
        help="Redis key prefix used for queue keys",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="Logging level",
    )

    return parser.parse_args()


def setup_signal_handlers(worker: Any, logger: logging.Logger) -> None:
    """Setup signal handlers for graceful shutdown."""

    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        if worker:
            worker.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


def main():
    """TaskWorker main entry point."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))

        logger = logging.getLogger(__name__)
        logger.info(f"Starting TaskWorker: {args.worker_id}")
        logger.info(
            "Configuration: redis=%s:%s db=%s prefix=%s",
            args.redis_host,
            args.redis_port,
            args.redis_db,
            args.redis_key_prefix,
        )

        redis_config = {
            "host": args.redis_host,
            "port": args.redis_port,
            "db": args.redis_db,
            "key_prefix": args.redis_key_prefix,
        }
        queue = create_redis_queue(redis_config, logger)

        # Create TaskWorker
        from graflow.worker.worker import TaskWorker

        worker = TaskWorker(
            queue=queue,
            worker_id=args.worker_id,
            max_concurrent_tasks=args.max_concurrent_tasks,
            poll_interval=args.poll_interval,
        )

        # Setup signal handlers
        setup_signal_handlers(worker, logger)

        # Start worker
        logger.info(f"TaskWorker {args.worker_id} starting...")
        worker.start()

        # Keep main thread alive
        try:
            while worker.is_running:
                time.sleep(1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, stopping worker...")
            worker.stop()

        logger.info(f"TaskWorker {args.worker_id} finished")

    except Exception as e:
        logger.error(f"TaskWorker failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
