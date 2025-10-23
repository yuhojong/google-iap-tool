import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
            store TEXT NOT NULL,
            sku TEXT NOT NULL,
            data TEXT NOT NULL,
            hash TEXT NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (store, sku)
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


def _update_last_sync(conn: sqlite3.Connection, store: str, timestamp: float) -> None:
    conn.execute(
        """
        INSERT INTO metadata(key, value)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (f"last_sync:{store}", str(timestamp)),
    )


def _load_cached_products(conn: sqlite3.Connection, store: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT data FROM products WHERE store = ? ORDER BY sku",
        (store,),
    ).fetchall()
    products: List[Dict[str, Any]] = []
    for row in rows:
        try:
            products.append(json.loads(row["data"]))
        except json.JSONDecodeError:
            logger.warning("Failed to decode cached product; skipping")
    return products


def _replace_products(conn: sqlite3.Connection, store: str, products: List[Dict[str, Any]]) -> None:
    existing_hashes = {
        row["sku"]: row["hash"]
        for row in conn.execute(
            "SELECT sku, hash FROM products WHERE store = ?",
            (store,),
        )
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
            INSERT INTO products(store, sku, data, hash, updated_at)
            VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(store, sku) DO UPDATE SET data=excluded.data, hash=excluded.hash, updated_at=excluded.updated_at
            """,
            (store, sku, _serialize_product(product), digest, now),
        )
    removed = set(existing_hashes.keys()) - set(incoming_hashes.keys())
    if removed:
        conn.executemany(
            "DELETE FROM products WHERE store = ? AND sku = ?",
            ((store, sku) for sku in removed),
        )
    _update_last_sync(conn, store, now)


def refresh_products_from_remote(
    store: str, fetch_remote: Callable[[], List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    with _lock:
        products = fetch_remote()
        with _get_connection() as conn:
            _replace_products(conn, store, products)
        logger.info("Refreshed product cache from remote", extra={"count": len(products)})
        return products


def get_cached_products(
    store: str,
    fetch_remote: Callable[[], List[Dict[str, Any]]],
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    with _lock:
        with _get_connection() as conn:
            cached = _load_cached_products(conn, store)
            if cached and not force_refresh:
                return cached
    return refresh_products_from_remote(store, fetch_remote)


def get_paginated_products(
    store: str,
    fetch_remote: Callable[[], List[Dict[str, Any]]],
    token: Optional[str],
    page_size: int = DEFAULT_PAGE_SIZE,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if page_size <= 0:
        raise ValueError("페이지 크기는 1 이상이어야 합니다.")
    if page_size > MAX_PAGE_SIZE:
        page_size = MAX_PAGE_SIZE

    products = get_cached_products(store, fetch_remote)
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


def upsert_product(store: str, product: Dict[str, Any]) -> None:
    sku = product.get("sku")
    if not sku:
        return
    with _lock:
        with _get_connection() as conn:
            now = time.time()
            conn.execute(
                """
                INSERT INTO products(store, sku, data, hash, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(store, sku) DO UPDATE SET data=excluded.data, hash=excluded.hash, updated_at=excluded.updated_at
                """,
                (store, sku, _serialize_product(product), _hash_product(product), now),
            )
            _update_last_sync(conn, store, now)


def delete_product(store: str, sku: str) -> None:
    if not sku:
        return
    with _lock:
        with _get_connection() as conn:
            conn.execute(
                "DELETE FROM products WHERE store = ? AND sku = ?",
                (store, sku),
            )
            _update_last_sync(conn, store, time.time())


def get_last_sync_time(store: str) -> Optional[float]:
    with _lock:
        with _get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM metadata WHERE key = ?",
                (f"last_sync:{store}",),
            ).fetchone()
            if not row:
                return None
            try:
                return float(row["value"])
            except (TypeError, ValueError):
                return None
