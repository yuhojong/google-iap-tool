import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from google_play import get_all_inapp_products

logger = logging.getLogger(__name__)

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 200
_DB_PATH = os.getenv("PRODUCT_CACHE_DB_PATH", os.path.join(os.getcwd(), "iap_cache.db"))
_lock = threading.Lock()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS products (
            sku TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            hash TEXT NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    return conn


def _serialize_product(product: Dict[str, Any]) -> str:
    return json.dumps(product, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _hash_product(product: Dict[str, Any]) -> str:
    payload = _serialize_product(product)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _update_last_sync(conn: sqlite3.Connection, timestamp: float) -> None:
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        ("last_sync", str(timestamp)),
    )


def _load_cached_products(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    rows = conn.execute("SELECT data FROM products ORDER BY sku").fetchall()
    products: List[Dict[str, Any]] = []
    for row in rows:
        try:
            products.append(json.loads(row["data"]))
        except json.JSONDecodeError:
            logger.warning("Failed to decode cached product; skipping")
    return products


def _replace_products(conn: sqlite3.Connection, products: List[Dict[str, Any]]) -> None:
    existing_hashes = {
        row["sku"]: row["hash"]
        for row in conn.execute("SELECT sku, hash FROM products")
    }
    incoming_hashes = {}
    now = time.time()
    for product in products:
        sku = product.get("sku")
        if not sku:
            continue
        digest = _hash_product(product)
        incoming_hashes[sku] = digest
        if existing_hashes.get(sku) == digest:
            continue
        conn.execute(
            """
            INSERT INTO products(sku, data, hash, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET data=excluded.data, hash=excluded.hash, updated_at=excluded.updated_at
            """,
            (sku, _serialize_product(product), digest, now),
        )
    removed = set(existing_hashes.keys()) - set(incoming_hashes.keys())
    if removed:
        conn.executemany("DELETE FROM products WHERE sku = ?", ((sku,) for sku in removed))
    _update_last_sync(conn, now)


def refresh_products_from_remote() -> List[Dict[str, Any]]:
    with _lock:
        products = get_all_inapp_products()
        with _get_connection() as conn:
            _replace_products(conn, products)
        logger.info("Refreshed product cache from remote", extra={"count": len(products)})
        return products


def get_cached_products(force_refresh: bool = False) -> List[Dict[str, Any]]:
    with _lock:
        with _get_connection() as conn:
            cached = _load_cached_products(conn)
            if cached and not force_refresh:
                return cached
    return refresh_products_from_remote()


def get_paginated_products(
    token: Optional[str], page_size: int = DEFAULT_PAGE_SIZE
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if page_size <= 0:
        raise ValueError("페이지 크기는 1 이상이어야 합니다.")
    if page_size > MAX_PAGE_SIZE:
        page_size = MAX_PAGE_SIZE

    products = get_cached_products()
    offset = 0
    if token:
        if not token.startswith("offset:"):
            raise ValueError("잘못된 페이지 토큰입니다.")
        try:
            offset = int(token.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError("잘못된 페이지 토큰입니다.") from exc
        if offset < 0:
            offset = 0
    page = products[offset : offset + page_size]
    next_offset = offset + page_size
    next_token = f"offset:{next_offset}" if next_offset < len(products) else None
    return page, next_token


def upsert_product(product: Dict[str, Any]) -> None:
    sku = product.get("sku")
    if not sku:
        return
    with _lock:
        with _get_connection() as conn:
            now = time.time()
            conn.execute(
                """
                INSERT INTO products(sku, data, hash, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET data=excluded.data, hash=excluded.hash, updated_at=excluded.updated_at
                """,
                (sku, _serialize_product(product), _hash_product(product), now),
            )
            _update_last_sync(conn, now)


def delete_product(sku: str) -> None:
    if not sku:
        return
    with _lock:
        with _get_connection() as conn:
            conn.execute("DELETE FROM products WHERE sku = ?", (sku,))
            _update_last_sync(conn, time.time())


def get_last_sync_time() -> Optional[float]:
    with _lock:
        with _get_connection() as conn:
            row = conn.execute("SELECT value FROM metadata WHERE key = ?", ("last_sync",)).fetchone()
            if not row:
                return None
            try:
                return float(row["value"])
            except (TypeError, ValueError):
                return None
