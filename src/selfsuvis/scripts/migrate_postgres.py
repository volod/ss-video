#!/usr/bin/env python3
"""Apply video and fusion PostgreSQL schemas.

Owners may share one Postgres instance. Video uses DATABASE_URL (database
``selfsuvis`` by default). Fusion uses FUSION_DATABASE_URL (database
``selfsuvis_fusion`` by default).

    python -m selfsuvis.scripts.migrate_postgres
    python -m selfsuvis.scripts.migrate_postgres --owner video
    python -m selfsuvis.scripts.migrate_postgres --owner fusion
"""

import argparse
import asyncio

from selfsuvis.pipeline.core.db_urls import sibling_database_url
from selfsuvis.pipeline.core.env import env_str, load_script_env


async def migrate(
    url: str,
    *,
    verbose: bool = True,
    owner: str = "all",
    fusion_url: str | None = None,
) -> None:
    """Apply the requested owner schema(s). ``url`` is the video DATABASE_URL."""
    if owner not in {"all", "video", "fusion"}:
        raise ValueError(f"unknown owner: {owner}")
    if owner in {"all", "video"}:
        from selfsuvis.pipeline.storage.migrate_video import migrate as migrate_video

        await migrate_video(url, verbose=verbose)
    if owner in {"all", "fusion"}:
        from selfsuvis.fusion_rt.migrate import migrate as migrate_fusion

        target = fusion_url or sibling_database_url(url, "selfsuvis_fusion")
        await migrate_fusion(target, verbose=verbose)


def main() -> None:
    load_script_env(anchor_file=__file__)
    parser = argparse.ArgumentParser(description="Apply video and/or fusion PostgreSQL schemas")
    parser.add_argument(
        "--owner",
        choices=("all", "video", "fusion"),
        default="all",
        help="Which service schema to apply (default: all)",
    )
    args = parser.parse_args()
    url = env_str(
        "DATABASE_URL",
        "postgresql://selfsuvis:selfsuvis@localhost:5432/selfsuvis",
    )
    fusion_url = env_str("FUSION_DATABASE_URL", "") or None
    asyncio.run(migrate(url, owner=args.owner, fusion_url=fusion_url))


if __name__ == "__main__":
    main()
