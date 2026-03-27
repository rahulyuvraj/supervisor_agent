"""Checkpointer factory — env-var gated for dev/prod flexibility."""
import os
import logging

logger = logging.getLogger(__name__)


def get_checkpointer():
    backend = os.getenv("CHECKPOINTER_BACKEND", "memory")

    if backend == "postgres":
        from langgraph.checkpoint.postgres import PostgresSaver
        conn_string = os.getenv("CHECKPOINTER_DB_URL")
        if not conn_string:
            raise RuntimeError(
                "CHECKPOINTER_BACKEND=postgres but CHECKPOINTER_DB_URL is not set"
            )
        saver = PostgresSaver.from_conn_string(conn_string)
        saver.setup()
        logger.info("Checkpointer: PostgreSQL (%s)", conn_string.split("@")[-1])
        return saver

    from langgraph.checkpoint.memory import MemorySaver
    logger.info("Checkpointer: in-memory (dev mode)")
    return MemorySaver()
