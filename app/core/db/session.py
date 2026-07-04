from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

async_engine = create_async_engine(
    settings.postgres_async_dsn,
    pool_pre_ping=True,
)
async_session_factory = async_sessionmaker(
    async_engine,
    expire_on_commit=False,
)

sync_engine = create_engine(
    settings.postgres_sync_dsn,
    pool_pre_ping=True,
)
sync_session_factory = sessionmaker(
    sync_engine,
    expire_on_commit=False,
)
