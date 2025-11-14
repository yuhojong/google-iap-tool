import asyncio
import base64
import binascii
import csv
import hashlib
import hmac
import io
import json
import logging
import mimetypes
import os
import posixpath
import shlex
import subprocess
import zipfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Set, Tuple

from contextlib import asynccontextmanager
from functools import wraps
import threading

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from dotenv import load_dotenv
from datetime import datetime, date

from apple_store import (
    AppleStoreConfigError,
    AppleStorePermissionError,
    create_inapp_purchase as create_apple_inapp_purchase,
    delete_inapp_purchase as delete_apple_inapp_purchase,
    remove_inapp_purchase_from_sale as remove_apple_inapp_purchase_from_sale,
    get_all_inapp_purchases,
    get_inapp_purchase_detail as get_apple_inapp_purchase_detail,
    get_inapp_purchase_ids_lightweight,
    get_fixed_price_territories,
    list_price_tiers as list_apple_price_tiers,
    get_iap_price_krw,
    setup_interrupt_handler,
    update_inapp_purchase as update_apple_inapp_purchase,
)
from google_play import (
    create_onetime_product,
    delete_inapp_product,
    get_all_google_play_products,
    update_managed_inapp,
)
from price_templates import (
    PriceTemplate,
    generate_price_templates_from_products,
    get_template_by_id,
    index_templates_by_price_micros,
)
from product_cache import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    delete_product as delete_cached_product,
    get_cached_products,
    get_cached_products_only,
    get_paginated_products,
    refresh_products_from_remote,
    upsert_product as upsert_cached_product,
    get_metadata_value,
    set_metadata_value,
)

load_dotenv()


class DailyLogFileHandler(logging.Handler):
    def __init__(self, directory: Path, encoding: str = "utf-8"):
        super().__init__()
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.encoding = encoding
        self._current_date: Optional[date] = None
        self._stream: Optional[Any] = None
        self._lock = threading.Lock()

    def _log_path_for(self, date_obj: date) -> Path:
        return self.directory / f"server_{date_obj.strftime('%Y-%m-%d')}.log"

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            today = datetime.now().date()
            with self._lock:
                if today != self._current_date:
                    self._current_date = today
                    if self._stream:
                        self._stream.close()
                    log_path = self._log_path_for(today)
                    self._stream = open(log_path, "a", encoding=self.encoding)
                if self._stream:
                    self._stream.write(msg + "\n")
                    self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        try:
            with self._lock:
                if self._stream:
                    self._stream.close()
                    self._stream = None
        finally:
            super().close()


log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
root_logger = logging.getLogger()
root_logger.setLevel(getattr(logging, log_level_name, logging.INFO))
root_logger.handlers.clear()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(stream_handler)

log_directory = Path(__file__).resolve().parent / "logs"
file_handler = DailyLogFileHandler(log_directory)
file_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

# Setup interrupt handler for graceful shutdown
setup_interrupt_handler()

GOOGLE_STORE = "google"
APPLE_STORE = "apple"
PROTECTED_TIMEOUT = 300  # seconds between refresh attempts (unused placeholder)
APPLE_PROTECTED_KEY = f"protected:{APPLE_STORE}"


def _parse_locale_list(value: str | None) -> List[str]:
    if not value:
        return []
    locales: List[str] = []
    for token in value.split(","):
        candidate = token.strip()
        if not candidate:
            continue
        if candidate not in locales:
            locales.append(candidate)
    return locales


APPLE_LOCALIZATION_LOCALES = _parse_locale_list(os.getenv("APPLE_LOCALIZATION_LOCALES"))
if not APPLE_LOCALIZATION_LOCALES:
    APPLE_LOCALIZATION_LOCALES = ["en-GB"]
elif "en-GB" not in APPLE_LOCALIZATION_LOCALES:
    APPLE_LOCALIZATION_LOCALES.insert(0, "en-GB")

APPLE_DEFAULT_LOCALIZATION_LOCALE = APPLE_LOCALIZATION_LOCALES[0]

# Apple debug mode: limit IAP fetch to 400 items when enabled
APPLE_DEBUG_MODE = os.getenv("APPLE_DEBUG_MODE", "false").lower() in ("true", "1", "yes")
APPLE_DEBUG_MAX_ITEMS = 400

app = FastAPI(title="iap-management-tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Handle Chrome DevTools well-known requests silently
@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools():
    """Chrome DevTools metadata endpoint (not used by this app)."""
    return {}

static_files = StaticFiles(directory="static", html=True)

app.mount("/static", static_files, name="static")


CSV_PROCESSING_LOCK = asyncio.Lock()
CSV_IDLE_EVENT = asyncio.Event()
CSV_IDLE_EVENT.set()
_csv_processing_counter = 0

DEPLOYMENT_LOCK = asyncio.Lock()


async def _run_in_thread(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

DEFAULT_REPO_PATH = Path(__file__).resolve().parent
TARGET_BRANCH = os.getenv("DEPLOY_TARGET_BRANCH", "main")
REPO_PATH = Path(os.getenv("DEPLOY_REPO_PATH", DEFAULT_REPO_PATH)).resolve()
RESTART_COMMAND = os.getenv("DEPLOY_RESTART_COMMAND")
POST_UPDATE_COMMANDS = os.getenv("DEPLOY_POST_UPDATE_COMMANDS")
GITHUB_WEBHOOK_SECRET = os.getenv("GITHUB_WEBHOOK_SECRET")


def _parse_additional_commands(value: str | None) -> List[List[str]]:
    commands: List[List[str]] = []
    if not value:
        return commands
    for line in value.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        commands.append(shlex.split(candidate))
    return commands


@asynccontextmanager
async def _csv_processing_guard():
    global _csv_processing_counter
    async with CSV_PROCESSING_LOCK:
        _csv_processing_counter += 1
        CSV_IDLE_EVENT.clear()
    try:
        yield
    finally:
        async with CSV_PROCESSING_LOCK:
            _csv_processing_counter = max(0, _csv_processing_counter - 1)
            if _csv_processing_counter == 0:
                CSV_IDLE_EVENT.set()


def csv_processing_endpoint(endpoint):
    @wraps(endpoint)
    async def wrapper(*args, **kwargs):
        async with _csv_processing_guard():
            return await endpoint(*args, **kwargs)

    return wrapper


async def _run_command(command: List[str], *, cwd: Path) -> None:
    command_display = " ".join(command)

    def _execute():
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout:
            logger.info("%s stdout: %s", command_display, result.stdout.strip())
        if result.stderr:
            logger.warning("%s stderr: %s", command_display, result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(
                f"Command '{command_display}' failed with exit code {result.returncode}"
            )

    logger.info("Running command: %s", command_display)
    await asyncio.to_thread(_execute)


async def _perform_deployment(trigger_id: str | None) -> None:
    async with DEPLOYMENT_LOCK:
        if trigger_id:
            logger.info("Deployment triggered by %s", trigger_id)
        else:
            logger.info("Deployment triggered")

        await CSV_IDLE_EVENT.wait()
        logger.info("CSV processing complete. Proceeding with deployment.")

        if not REPO_PATH.exists():
            raise RuntimeError(f"Repository path does not exist: {REPO_PATH}")

        commands: List[List[str]] = [
            ["git", "fetch", "origin", TARGET_BRANCH],
            ["git", "reset", "--hard", f"origin/{TARGET_BRANCH}"],
        ]
        commands.extend(_parse_additional_commands(POST_UPDATE_COMMANDS))

        if RESTART_COMMAND:
            commands.append(shlex.split(RESTART_COMMAND))
        else:
            logger.warning(
                "DEPLOY_RESTART_COMMAND is not set. Deployment will not restart the service."
            )

        for command in commands:
            await _run_command(command, cwd=REPO_PATH)

        logger.info("Deployment finished successfully.")


def _verify_github_signature(secret: str, payload: bytes, signature_header: str | None) -> bool:
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    signature = signature_header.split("=", 1)[1]
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, signature)


def _schedule_deployment(trigger_id: str | None) -> None:
    asyncio.create_task(_perform_deployment(trigger_id))


@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    if not GITHUB_WEBHOOK_SECRET:
        logger.error("GITHUB_WEBHOOK_SECRET is not configured.")
        raise HTTPException(status_code=500, detail="Webhook secret is not configured.")

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not _verify_github_signature(GITHUB_WEBHOOK_SECRET, body, signature):
        logger.warning("Received GitHub webhook with invalid signature.")
        raise HTTPException(status_code=401, detail="Invalid webhook signature.")

    event = request.headers.get("X-GitHub-Event", "")
    if event != "push":
        logger.info("Ignoring GitHub event: %s", event)
        return {"status": "ignored", "reason": "unsupported_event"}

    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode webhook payload: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc

    expected_ref = f"refs/heads/{TARGET_BRANCH}"
    ref = payload.get("ref")
    if ref != expected_ref:
        logger.info("Ignoring push to %s. Watching %s.", ref, expected_ref)
        return {"status": "ignored", "reason": "branch_mismatch"}

    delivery = request.headers.get("X-GitHub-Delivery")
    head_commit = payload.get("after") or (payload.get("head_commit") or {}).get("id")
    trigger_id = head_commit or delivery

    background_tasks.add_task(_schedule_deployment, trigger_id)

    return {"status": "queued"}


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


class Translation(BaseModel):
    language: str = Field(..., min_length=2)
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)


class CreateInAppRequest(BaseModel):
    sku: str = Field(..., min_length=1)
    default_language: str = Field("ko-KR", min_length=2)
    price_won: int | None = Field(default=None, ge=100)
    price_template_id: str | None = Field(default=None, min_length=1)
    translations: list[Translation] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_pricing(cls, model: "CreateInAppRequest"):
        translations = model.translations or []
        languages = {t.language for t in translations}
        default_language = model.default_language
        if default_language not in languages:
            raise ValueError("기본 언어에 대한 번역 정보를 입력해야 합니다.")

        has_manual_price = model.price_won is not None
        has_template = model.price_template_id is not None
        if has_manual_price == has_template:
            raise ValueError("가격 템플릿 또는 직접 입력 가격 중 하나만 선택해야 합니다.")
        return model


class ListingPayload(BaseModel):
    title: str
    description: str


class PricePayload(BaseModel):
    priceMicros: str
    currency: str


class ImportProductPayload(BaseModel):
    sku: str
    status: str = Field(default="active")
    default_language: str
    default_price: PricePayload
    listings: Dict[str, ListingPayload]
    prices: Optional[Dict[str, Any]] = None

    @field_validator("listings")
    @classmethod
    def ensure_default_language_listing(
        cls, listings: Dict[str, ListingPayload], info: Dict[str, Any]
    ) -> Dict[str, ListingPayload]:
        default_language = info.data.get("default_language")
        if default_language and default_language not in listings:
            raise ValueError("기본 언어 번역 정보가 필요합니다.")
        return listings


class UpdateInAppRequest(BaseModel):
    default_language: str = Field(..., min_length=2)
    status: str = Field(default="active", min_length=1)
    default_price: PricePayload
    listings: Dict[str, ListingPayload]
    prices: Optional[Dict[str, PricePayload]] = None

    @field_validator("listings")
    @classmethod
    def ensure_default_language_listing(
        cls, listings: Dict[str, ListingPayload], info: Dict[str, Any]
    ) -> Dict[str, ListingPayload]:
        default_language = info.data.get("default_language")
        if default_language and default_language not in listings:
            raise ValueError("기본 언어 번역 정보가 필요합니다.")
        return listings


class ImportOperation(BaseModel):
    action: Literal["create", "update", "delete"]
    sku: str
    data: Optional[ImportProductPayload] = None


class ImportApplyRequest(BaseModel):
    operations: List[ImportOperation]


class BulkCreateOperation(BaseModel):
    action: Literal["create"] = Field(default="create")
    sku: str
    data: ImportProductPayload


class BulkCreateApplyRequest(BaseModel):
    operations: List[BulkCreateOperation]


class AppleReviewScreenshot(BaseModel):
    filename: str = Field(..., min_length=1)
    content_type: str = Field(..., min_length=1)
    data: str = Field(..., min_length=1)

    @field_validator("data")
    @classmethod
    def validate_base64(cls, value: str) -> str:
        try:
            base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError("스크린샷 데이터는 base64 문자열이어야 합니다.") from exc
        return value

    def decode_bytes(self) -> bytes:
        return base64.b64decode(self.data)


class AppleLocalization(BaseModel):
    locale: str = Field(..., min_length=2)
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    review_screenshot: Optional[AppleReviewScreenshot] = None


class AppleCreateInAppRequest(BaseModel):
    product_id: str = Field(..., min_length=1)
    reference_name: str = Field(..., min_length=1)
    purchase_type: Literal[
        "consumable",
        "nonConsumable",
        "nonRenewingSubscription",
        "autoRenewableSubscription",
    ]
    cleared_for_sale: bool = Field(default=True)
    family_sharable: bool = Field(default=False)
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_krw: int = Field(..., ge=0)
    price_point_id: Optional[str] = Field(default=None)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: List[AppleLocalization] = Field(default_factory=list)

    @model_validator(mode="after")
    def ensure_localizations(cls, model: "AppleCreateInAppRequest"):
        if not model.localizations:
            raise ValueError("최소 1개의 현지화 정보를 입력해야 합니다.")
        return model


class AppleUpdateInAppRequest(BaseModel):
    reference_name: Optional[str] = Field(default=None, min_length=1)
    cleared_for_sale: Optional[bool] = None
    family_sharable: Optional[bool] = None
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_krw: Optional[int] = Field(default=None, ge=0)
    price_point_id: Optional[str] = Field(default=None)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: Dict[str, AppleLocalization] = Field(default_factory=dict)


class AppleBulkDeletePreviewRequest(BaseModel):
    identifier_type: Literal["reference_name", "product_id", "sku", "iap_id"]
    values: List[str] = Field(..., min_length=1)

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value):
        if isinstance(value, str):
            tokens = [token.strip() for token in value.replace(",", "\n").splitlines()]
            return [token for token in tokens if token]
        if isinstance(value, list):
            result = []
            for item in value:
                if isinstance(item, str):
                    token = item.strip()
                    if token:
                        result.append(token)
            return result
        raise TypeError("values must be a string or list of strings")

    @model_validator(mode="after")
    def _ensure_values(self):  # type: ignore[override]
        unique = list(dict.fromkeys(self.values))
        if not unique:
            raise ValueError("삭제할 항목을 하나 이상 입력해주세요.")
        object.__setattr__(self, "values", unique)
        return self


class AppleBulkDeleteItem(BaseModel):
    inapp_id: str = Field(..., min_length=1)
    product_id: str = Field(..., min_length=1)

    @model_validator(mode="after")
    def _normalize(self):  # type: ignore[override]
        object.__setattr__(self, "inapp_id", self.inapp_id.strip())
        object.__setattr__(self, "product_id", self.product_id.strip())
        if not self.inapp_id or not self.product_id:
            raise ValueError("유효한 상품 정보를 입력해주세요.")
        return self


class AppleBulkDeleteApplyRequest(BaseModel):
    items: List[AppleBulkDeleteItem] = Field(..., min_length=1)


class AppleProtectionRequest(BaseModel):
    product_ids: List[str] = Field(..., min_length=1)


class AppleImportPayload(BaseModel):
    product_id: str = Field(..., min_length=1)
    reference_name: str = Field(..., min_length=1)
    purchase_type: Optional[
        Literal[
            "consumable",
            "nonConsumable",
            "nonRenewingSubscription",
            "autoRenewableSubscription",
        ]
    ] = None
    cleared_for_sale: bool = Field(default=True)
    family_sharable: bool = Field(default=False)
    review_note: Optional[str] = Field(default=None, max_length=4000)
    price_krw: Optional[int] = Field(default=None, ge=0)
    base_territory: str = Field(default="KOR", min_length=2)
    localizations: List[AppleLocalization] = Field(default_factory=list)


class AppleImportOperation(BaseModel):
    action: Literal["create", "update", "delete"]
    product_id: str = Field(..., min_length=1)
    data: Optional[AppleImportPayload] = None


class AppleImportApplyRequest(BaseModel):
    operations: List[AppleImportOperation]


def _collect_languages_from_products(products: Iterable[Dict[str, Any]]) -> List[str]:
    languages: set[str] = set()
    for item in products:
        listings = item.get("listings") or {}
        if isinstance(listings, dict):
            for language in listings.keys():
                if isinstance(language, str):
                    languages.add(language)
        elif isinstance(listings, list):
            for entry in listings:
                if not isinstance(entry, dict):
                    continue
                language = entry.get("languageCode") or entry.get("language")
                if isinstance(language, str):
                    languages.add(language)
    return sorted(languages)


def _collect_billable_regions_from_products(
    products: Iterable[Dict[str, Any]]
) -> Set[str]:
    regions: Set[str] = set()
    for item in products:
        if not isinstance(item, dict):
            continue
        canonical = (
            item
            if "product_type" in item and isinstance(item.get("prices"), (dict, type(None)))
            else _canonicalize_google_product(item)
        )
        if not isinstance(canonical, dict):
            continue
        price_map = canonical.get("prices")
        if isinstance(price_map, dict):
            for region_code, payload in price_map.items():
                if not isinstance(region_code, str):
                    continue
                if isinstance(payload, dict):
                    price_micros = payload.get("priceMicros")
                    currency = payload.get("currency")
                    if price_micros and currency:
                        regions.add(region_code.strip().upper())
    return regions


def _extract_billable_regions_from_product(product: Dict[str, Any]) -> Set[str]:
    if not isinstance(product, dict):
        return set()
    canonical = (
        product
        if "product_type" in product and isinstance(product.get("prices"), dict)
        else _canonicalize_google_product(product)
    )
    if not isinstance(canonical, dict):
        return set()
    price_map = canonical.get("prices")
    if not isinstance(price_map, dict):
        return set()
    regions: Set[str] = set()
    for region_code, payload in price_map.items():
        if not isinstance(region_code, str):
            continue
        if isinstance(payload, dict):
            price_micros = payload.get("priceMicros")
            currency = payload.get("currency")
            if price_micros and currency:
                regions.add(region_code.strip().upper())
    return regions


def _paginate_cached_products(
    products: List[Dict[str, Any]], token: Optional[str], page_size: int
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    if page_size <= 0:
        raise ValueError("페이지 크기는 1 이상이어야 합니다.")
    if page_size > MAX_PAGE_SIZE:
        page_size = MAX_PAGE_SIZE

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


def _canonicalize_google_product(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    source = item.get("__source") or ""
    product_type = {
        "monetization_onetime": "onetime",
        "monetization_subscription": "subscription",
    }.get(source, "unknown")

    def _resolve_sku() -> str:
        candidates = [
            item.get("sku"),
            item.get("productId"),
            item.get("oneTimeProductId"),
            item.get("subscriptionId"),
        ]
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        name = item.get("name")
        if isinstance(name, str) and name:
            last_segment = name.split("/")[-1]
            if last_segment:
                return last_segment
        return ""

    def _iter_listings():
        listings_obj = item.get("listings")
        if isinstance(listings_obj, dict) and listings_obj:
            for language, payload in listings_obj.items():
                yield language, payload
            return
        if isinstance(listings_obj, list) and listings_obj:
            for entry in listings_obj:
                if not isinstance(entry, dict):
                    continue
                language = entry.get("languageCode") or entry.get("language")
                if not isinstance(language, str) or not language:
                    continue
                yield language, entry

        localized_resources = item.get("localizedResources") or item.get("localized_resources")
        if isinstance(localized_resources, list):
            for payload in localized_resources:
                if not isinstance(payload, dict):
                    continue
                language = payload.get("languageCode") or payload.get("language")
                if not isinstance(language, str) or not language:
                    continue
                yield language, {
                    "title": payload.get("title") or payload.get("name"),
                    "description": payload.get("description") or payload.get("shortDescription"),
                }

    first_listing_language: Optional[str] = None
    normalized_listings: Dict[str, Dict[str, str]] = {}
    for language, listing in _iter_listings() or []:
        if not isinstance(language, str) or not isinstance(listing, dict):
            continue
        title = (listing.get("title") or listing.get("name") or "").strip()
        description = (listing.get("description") or listing.get("shortDescription") or "").strip()
        if not title and not description:
            continue
        if not first_listing_language:
            first_listing_language = language
        normalized_listings[language] = {
            "title": title or "",
            "description": description or "",
        }

    def _normalize_price_entry(price_obj: Any) -> Optional[Dict[str, str]]:
        if not isinstance(price_obj, dict):
            return None

        if "priceMicros" in price_obj and "currency" in price_obj:
            price_micros = price_obj.get("priceMicros")
            currency = price_obj.get("currency")
            return {
                "priceMicros": str(price_micros or ""),
                "currency": currency or "",
            }

        money = price_obj.get("price") if isinstance(price_obj.get("price"), dict) else price_obj
        if not isinstance(money, dict):
            return None
        currency_code = money.get("currencyCode")
        if not currency_code:
            return None
        units = money.get("units", "0")
        nanos = money.get("nanos", 0)
        try:
            units_int = int(str(units))
            nanos_int = int(str(nanos))
        except ValueError:
            logger.debug("Failed to parse money value: %s", money)
            return None
        price_micros = units_int * 1_000_000 + nanos_int // 1_000
        return {"priceMicros": str(price_micros), "currency": currency_code}

    price_map: Dict[str, Dict[str, str]] = {}

    direct_prices = item.get("prices")
    if isinstance(direct_prices, dict):
        for region, payload in direct_prices.items():
            if isinstance(payload, dict) and "regionCode" in payload and "price" in payload:
                region_code = payload.get("regionCode")
                normalized = _normalize_price_entry(payload.get("price"))
            else:
                region_code = region
                normalized = _normalize_price_entry(payload)
            if isinstance(region_code, str) and normalized:
                price_map[region_code] = normalized

    for config in item.get("regionalConfigs", []) or []:
        if not isinstance(config, dict):
            continue
        region_code = config.get("regionCode")
        price_candidate: Any = config.get("price") or config.get("newestPrice") or config.get("oldestPrice")
        if isinstance(price_candidate, dict) and "price" in price_candidate:
            price_candidate = price_candidate.get("price")
        normalized = _normalize_price_entry(price_candidate)
        if isinstance(region_code, str) and normalized:
            price_map.setdefault(region_code, normalized)

    purchase_options = item.get("purchaseOptions")
    status: str = (item.get("status") or item.get("state") or "").strip()
    selected_purchase_option: Optional[Dict[str, Any]] = None
    if isinstance(purchase_options, list):
        for option in purchase_options:
            if not isinstance(option, dict):
                continue
            option_state = option.get("state")
            if option_state:
                status = str(option_state).strip()
            if selected_purchase_option is None:
                selected_purchase_option = option
            elif str(option.get("state", "")).strip().upper() == "ACTIVE" and str(
                selected_purchase_option.get("state", "")
            ).strip().upper() != "ACTIVE":
                selected_purchase_option = option
            configs = option.get("regionalPricingAndAvailabilityConfigs")
            if isinstance(configs, list):
                for config in configs:
                    if not isinstance(config, dict):
                        continue
                    availability = config.get("availability")
                    if isinstance(availability, str) and availability.upper() not in ("AVAILABLE", "AVAILABILITY_UNSPECIFIED"):
                        continue
                    region_code = config.get("regionCode")
                    normalized = _normalize_price_entry(config.get("price"))
                    if isinstance(region_code, str) and normalized:
                        price_map.setdefault(region_code, normalized)

    for base_plan in item.get("basePlans", []) or []:
        if not isinstance(base_plan, dict):
            continue
        plan_state = base_plan.get("state")
        if plan_state:
            status = plan_state.lower()
        regional_prices = (
            base_plan.get("regionalPrices")
            or base_plan.get("prices")
            or base_plan.get("regional_configs")
        )
        if isinstance(regional_prices, dict):
            for region, payload in regional_prices.items():
                if isinstance(payload, dict) and "regionCode" in payload and "price" in payload:
                    region_code = payload.get("regionCode")
                    normalized = _normalize_price_entry(payload.get("price"))
                else:
                    region_code = region
                    normalized = _normalize_price_entry(payload)
                if isinstance(region_code, str) and normalized:
                    price_map.setdefault(region_code, normalized)
        elif isinstance(regional_prices, list):
            for payload in regional_prices:
                if not isinstance(payload, dict):
                    continue
                region_code = payload.get("regionCode") or payload.get("region")
                price_candidate: Any = payload.get("price") or payload.get("priceAmount") or payload
                if isinstance(price_candidate, dict) and "price" in price_candidate:
                    price_candidate = price_candidate.get("price")
                normalized = _normalize_price_entry(price_candidate)
                if isinstance(region_code, str) and normalized:
                    price_map.setdefault(region_code, normalized)

    default_price_raw = item.get("defaultPrice")
    normalized_default_price = _normalize_price_entry(default_price_raw) if default_price_raw else None

    if not normalized_default_price and price_map:
        preferred_region = item.get("defaultRegionCode") or "KR"
        if isinstance(preferred_region, str) and preferred_region in price_map:
            normalized_default_price = dict(price_map[preferred_region])
        else:
            first_region = next(iter(price_map.values()))
            normalized_default_price = dict(first_region)

    normalized_prices: Dict[str, Dict[str, str]] = {}
    for region, price in price_map.items():
        normalized_prices[region] = {
            "priceMicros": price.get("priceMicros", ""),
            "currency": price.get("currency", ""),
        }

    default_language = (
        item.get("defaultLanguage")
        or item.get("defaultLanguageCode")
        or item.get("defaultLanguageTag")
        or first_listing_language
        or ""
    )

    sku = _resolve_sku()
    if not sku:
        logger.debug("Skipping Google Play product without identifiable SKU: %s", item.get("name"))
        return None

    def _normalize_status(raw_status: str) -> str:
        value = (raw_status or "").strip().upper()
        if value in {"ACTIVE", "ON_SALE", "AVAILABLE"}:
            return "active"
        if value in {"INACTIVE", "STOPPED", "ARCHIVED", "DISABLED"}:
            return "inactive"
        if value in {"DRAFT", "PENDING"}:
            return "draft"
        return value.lower() if value else "unknown"

    def _extract_numeric_price(price_obj: Optional[Dict[str, str]]) -> Tuple[Optional[int], Optional[str]]:
        if not isinstance(price_obj, dict):
            return (None, None)
        price_micros = price_obj.get("priceMicros")
        currency = price_obj.get("currency")
        if price_micros is None or currency is None:
            return (None, None)
        try:
            return (int(price_micros), str(currency))
        except (TypeError, ValueError):
            return (None, None)

    default_price_micros, default_price_currency = _extract_numeric_price(normalized_default_price)

    available_regions = {
        region
        for region, price in price_map.items()
        if price.get("currency") and price.get("priceMicros")
    }

    should_skip = False
    if default_price_micros is None or default_price_currency is None:
        should_skip = True
    if normalized_prices and not available_regions:
        should_skip = True

    if should_skip:
        logger.info(
            "Skipping Google Play product '%s' without usable price information (likely loyalty/promotional item)",
            sku,
        )
        return None

    normalized_default_price = {
        "priceMicros": str(default_price_micros),
        "currency": default_price_currency,
    }

    return {
        "sku": sku,
        "status": _normalize_status(status),
        "default_language": default_language,
        "default_price": normalized_default_price,
        "listings": normalized_listings,
        "prices": normalized_prices or None,
        "product_type": product_type,
        "__source": source,
        "purchase_option_id": (selected_purchase_option or {}).get("purchaseOptionId"),
    }


def _fetch_google_products() -> List[Dict[str, Any]]:
    raw_products = get_all_google_play_products()
    filtered: List[Dict[str, Any]] = []
    skipped = 0
    for product in raw_products:
        canonical = _canonicalize_google_product(product)
        if not canonical:
            skipped += 1
            continue
        filtered.append(product)
    if skipped:
        logger.info("Filtered out %d Google Play promotional products without price", skipped)
    return filtered


def _fetch_apple_products() -> List[Dict[str, Any]]:
    # Apply debug mode limit at fetch time to avoid fetching too many items
    max_items = APPLE_DEBUG_MAX_ITEMS if APPLE_DEBUG_MODE else None
    if max_items:
        logger.info(
            f"Apple debug mode enabled: limiting fetch to {max_items} items"
        )
    
    products, _ = get_all_inapp_purchases(
        include_relationships=True,
        max_items=max_items,
        summary_only=True,
    )

    return products


def _check_apple_products_changed(
    cached_products: List[Dict[str, Any]]
) -> Tuple[bool, Optional[set], Optional[set]]:
    """
    Check if Apple IAPs have changed since last fetch.
    Returns (needs_refresh, added_ids, removed_ids).
    If needs_refresh is False, added_ids and removed_ids are None.
    """
    try:
        # Get current IAP IDs from Apple
        current_ids, total_count = get_inapp_purchase_ids_lightweight()
        current_ids_set = set(current_ids)
        
        # Get cached IAP IDs
        cached_ids_set = set()
        for product in cached_products:
            product_id = product.get("productId") or product.get("sku") or ""
            if product_id:
                cached_ids_set.add(product_id)
        
        # Check if any IDs are different
        added = current_ids_set - cached_ids_set
        removed = cached_ids_set - current_ids_set
        
        if not added and not removed:
            logger.info(
                "Apple IAP list unchanged (%d items) - using cache",
                len(cached_ids_set)
            )
            return False, None, None
        
        logger.info(
            "Apple IAP list changed: %d added, %d removed",
            len(added), len(removed)
        )
        return True, added, removed
        
    except Exception as exc:
        logger.warning("Failed to check for Apple IAP changes: %s - will refresh", exc)
        return True, None, None  # On error, full refresh


def _collect_locales_from_apple_products(products: Iterable[Dict[str, Any]]) -> List[str]:
    locales: set[str] = set()
    for product in products:
        localization_map = product.get("localizations") or {}
        for locale in localization_map.keys():
            if isinstance(locale, str) and locale:
                locales.add(locale)
    return sorted(locales)


def _to_apple_summary(product: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(product, dict):
        return {}
    summary = {
        "id": product.get("id"),
        "resourceType": product.get("resourceType"),
        "sku": product.get("productId") or product.get("sku"),
        "productId": product.get("productId") or product.get("sku"),
        "referenceName": product.get("referenceName"),
        "type": product.get("type"),
        "state": product.get("state"),
    }

    krw_price = product.get("krwPrice")
    if krw_price:
        summary["krwPrice"] = krw_price
    price_tier = product.get("priceTier")
    if price_tier:
        summary["priceTier"] = price_tier
    return summary


def _normalize_zip_path(name: str) -> str:
    normalized = posixpath.normpath(name.replace("\\", "/")).strip("/")
    return normalized


def _detect_content_type(filename: str) -> str:
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed
    return "application/octet-stream"


def _extract_csv_bundle(content: bytes, filename: str | None) -> Tuple[bytes, Dict[str, Dict[str, Any]]]:
    if not filename or not filename.lower().endswith(".zip"):
        return content, {}

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            csv_candidates: List[Tuple[str, bytes]] = []
            attachments: Dict[str, Dict[str, Any]] = {}
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                entry_name = _normalize_zip_path(entry.filename)
                try:
                    data = archive.read(entry)
                except KeyError as exc:
                    raise HTTPException(status_code=400, detail=f"ZIP 파일에서 {entry.filename} 을(를) 읽을 수 없습니다.") from exc
                if entry_name.lower().endswith(".csv"):
                    csv_candidates.append((entry_name, data))
                    continue
                attachment = {
                    "filename": os.path.basename(entry_name) or entry_name,
                    "content_type": _detect_content_type(entry_name),
                    "data": data,
                }
                keys = {
                    entry_name,
                    entry_name.lower(),
                    os.path.basename(entry_name),
                    os.path.basename(entry_name).lower(),
                }
                for key in keys:
                    if key:
                        attachments.setdefault(key, attachment)
            if not csv_candidates:
                raise HTTPException(status_code=400, detail="ZIP 파일에서 CSV를 찾을 수 없습니다.")
            csv_candidates.sort(key=lambda item: (item[0].count("/"), len(item[0])))
            return csv_candidates[0][1], attachments
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="유효한 ZIP 파일이 아닙니다.") from exc

    raise HTTPException(status_code=400, detail="ZIP 파일을 처리할 수 없습니다.")


def _resolve_attachment(
    value: str,
    attachments: Dict[str, Dict[str, Any]] | None,
) -> Dict[str, Any]:
    if not value:
        raise HTTPException(status_code=400, detail="스크린샷 파일명을 입력해주세요.")
    if not attachments:
        raise HTTPException(
            status_code=400,
            detail=f"스크린샷 파일 '{value}' 을(를) 찾을 수 없습니다. ZIP 파일에 이미지를 포함했는지 확인해주세요.",
        )
    normalized = _normalize_zip_path(value)
    candidates = [
        normalized,
        normalized.lower(),
        os.path.basename(normalized),
        os.path.basename(normalized).lower(),
    ]
    for key in candidates:
        if key in attachments:
            return attachments[key]
    raise HTTPException(
        status_code=400,
        detail=f"스크린샷 파일 '{value}' 을(를) ZIP 파일에서 찾을 수 없습니다.",
    )


def _parse_bool_cell(value: str, *, default: bool = False) -> bool:
    text = (value or "").strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "y", "t"}:
        return True
    if text in {"0", "false", "no", "n", "f"}:
        return False
    raise ValueError("불리언 필드에는 true/false 값을 입력해주세요.")


def _normalize_price_won(value: str) -> tuple[str, str]:
    stripped = value.strip().replace(",", "").replace("_", "")
    if not stripped:
        raise ValueError("KRW 가격 값을 입력해주세요.")
    try:
        decimal_value = Decimal(stripped)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("KRW 가격은 숫자여야 합니다.") from exc
    if decimal_value <= 0:
        raise ValueError("KRW 가격은 0보다 커야 합니다.")
    scaled = decimal_value * Decimal("1000000")
    if scaled != scaled.to_integral_value():
        raise ValueError("KRW 가격은 소수점 여섯째 자리까지만 입력할 수 있습니다.")
    price_micros = scaled.to_integral_value()
    normalized_price = decimal_value.normalize()
    price_text = format(normalized_price, "f")
    return str(int(price_micros)), price_text


def _format_price_won_from_micros(price_micros: str, currency: str) -> str:
    if currency.upper() != "KRW":  # type: ignore[attr-defined]
        return ""
    try:
        decimal_value = Decimal(price_micros) / Decimal("1000000")
    except (InvalidOperation, ValueError):
        return ""
    text = f"{decimal_value:,.6f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _compute_changes(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    if before.get("status") != after.get("status"):
        changes["status"] = {"from": before.get("status"), "to": after.get("status")}
    if before.get("default_language") != after.get("default_language"):
        changes["default_language"] = {
            "from": before.get("default_language"),
            "to": after.get("default_language"),
        }

    before_price = before.get("default_price") or {}
    after_price = after.get("default_price") or {}
    if (
        before_price.get("priceMicros") != after_price.get("priceMicros")
        or before_price.get("currency") != after_price.get("currency")
    ):
        changes["default_price"] = {
            "from": before_price,
            "to": after_price,
        }

    listing_changes: Dict[str, Any] = {}
    before_listings = before.get("listings") or {}
    after_listings = after.get("listings") or {}
    languages = set(before_listings.keys()) | set(after_listings.keys())
    for language in sorted(languages):
        prev = before_listings.get(language)
        nxt = after_listings.get(language)
        if prev != nxt:
            listing_changes[language] = {"from": prev, "to": nxt}
    if listing_changes:
        changes["listings"] = listing_changes

    before_prices = before.get("prices") or {}
    after_prices = after.get("prices") or {}
    if before_prices != after_prices:
        changes["prices"] = {"from": before_prices, "to": after_prices}
    return changes


def _parse_import_csv(content: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    languages = sorted({name.split("title_", 1)[1] for name in reader.fieldnames if name.startswith("title_")})
    rows: List[Dict[str, Any]] = []
    seen_skus: set[str] = set()

    for index, row in enumerate(reader, start=2):
        sku = (row.get("sku") or "").strip()
        if not sku:
            continue
        if sku in seen_skus:
            raise HTTPException(status_code=400, detail=f"CSV에 중복된 SKU가 있습니다: {sku}")
        seen_skus.add(sku)

        entry: Dict[str, Any] = {
            "row": index,
            "sku": sku,
            "status": (row.get("status") or "").strip() or None,
            "default_language": (row.get("default_language") or "").strip() or None,
            "price_won": (row.get("price_won") or "").strip() or None,
            "listings": {},
            "prices": None,
        }
        for language in languages:
            title_key = f"title_{language}"
            description_key = f"description_{language}"
            title = (row.get(title_key) or "").strip()
            description = (row.get(description_key) or "").strip()
            if title or description:
                entry["listings"][language] = {
                    "title": title,
                    "description": description,
                }

        rows.append(entry)

    return rows, languages


def _parse_apple_import_csv(
    content: bytes, attachments: Dict[str, Dict[str, Any]] | None = None
) -> tuple[List[Dict[str, Any]], List[str]]:
    """Parse Apple IAP import CSV.
    
    Expected CSV format:
    - productId: 상품 ID (필수)
    - type: 타입 (필수)
    - price_krw: KRW 가격 (정수, 필수)
    - name_{locale}: 각 로케일별 제목 (APPLE_LOCALIZATION_LOCALES 기준)
    - description_{locale}: 각 로케일별 설명 (APPLE_LOCALIZATION_LOCALES 기준)
    """
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    # Use APPLE_LOCALIZATION_LOCALES as the expected locales
    expected_locales = APPLE_LOCALIZATION_LOCALES
    
    # Also collect any additional locales found in CSV (for backward compatibility)
    found_locales: set[str] = set()
    for name in reader.fieldnames:
        if name.startswith("name_"):
            found_locales.add(name.split("_", 1)[1])
        if name.startswith("description_"):
            found_locales.add(name.split("_", 1)[1])
        if name.startswith("screenshot_"):
            found_locales.add(name.split("_", 1)[1])
    
    # Use expected locales, but also include any found locales
    locales = sorted(set(expected_locales) | found_locales)

    rows: List[Dict[str, Any]] = []
    for index, row in enumerate(reader, start=2):
        normalized = {key.strip(): (value or "").strip() for key, value in (row or {}).items()}
        normalized["__row"] = str(index)
        
        # Validate required fields
        product_id = normalized.get("productId") or normalized.get("product_id") or ""
        product_type = normalized.get("type") or ""
        price_krw = normalized.get("price_krw") or ""
        
        if not product_id:
            raise HTTPException(status_code=400, detail=f"행 {index}: productId가 필요합니다.")
        if not product_type:
            raise HTTPException(status_code=400, detail=f"행 {index}: type이 필요합니다.")
        if not price_krw:
            raise HTTPException(status_code=400, detail=f"행 {index}: price_krw가 필요합니다.")
        
        # Normalize field names
        if "product_id" in normalized and "productId" not in normalized:
            normalized["productId"] = normalized["product_id"]
        
        # Handle screenshots (optional)
        screenshots: Dict[str, Dict[str, Any]] = {}
        for locale in locales:
            key = f"screenshot_{locale}"
            screenshot_value = normalized.get(key, "")
            if not screenshot_value:
                continue
            if not attachments:
                continue
            attachment = _resolve_attachment(screenshot_value, attachments)
            data_bytes = attachment["data"]
            if len(data_bytes) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {index}: 스크린샷 '{screenshot_value}' 용량이 10MB를 초과합니다.",
                )
            content_type = attachment.get("content_type") or "application/octet-stream"
            if content_type not in {"image/png", "image/jpeg"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {index}: 스크린샷 '{screenshot_value}' 은 PNG 또는 JPG 파일이어야 합니다.",
                )
            encoded = base64.b64encode(data_bytes).decode("ascii")
            screenshots[locale] = {
                "filename": attachment.get("filename") or screenshot_value,
                "content_type": content_type,
                "data": encoded,
            }
        if screenshots:
            normalized["__screenshots"] = screenshots
        rows.append(normalized)

    return rows, locales


def _parse_apple_batch_create_csv(content: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
    """Parse CSV for Apple batch IAP creation (only new IAPs)."""
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    # Extract locales from column headers (name_en-US, description_ko, etc.)
    locales = sorted({
        name.split("_", 1)[1] 
        for name in reader.fieldnames 
        if (name.startswith("name_") or name.startswith("description_")) and "_" in name
    })
    
    rows: List[Dict[str, Any]] = []
    seen_product_ids: set[str] = set()

    for index, row in enumerate(reader, start=2):
        product_id = (row.get("product_id") or "").strip()
        reference_name = (row.get("reference_name") or "").strip()
        iap_type = (row.get("type") or "").strip()
        price_tier = (row.get("price_tier") or "").strip()

        # Skip empty rows
        if not product_id and not reference_name:
            continue

        if not product_id:
            raise HTTPException(status_code=400, detail=f"{index} 행: product_id 값을 입력해주세요.")
        if product_id in seen_product_ids:
            raise HTTPException(status_code=400, detail=f"CSV에 중복된 Product ID가 있습니다: {product_id}")
        seen_product_ids.add(product_id)

        if not reference_name:
            raise HTTPException(status_code=400, detail=f"{index} 행: reference_name 값을 입력해주세요.")
        if not iap_type:
            raise HTTPException(status_code=400, detail=f"{index} 행: type 값을 입력해주세요.")
        if not price_tier:
            raise HTTPException(status_code=400, detail=f"{index} 행: price_tier 값을 입력해주세요.")

        # Parse localizations
        localizations: Dict[str, Dict[str, str]] = {}
        for locale in locales:
            name_key = f"name_{locale}"
            description_key = f"description_{locale}"
            name = (row.get(name_key) or "").strip()
            description = (row.get(description_key) or "").strip()
            if name and description:
                localizations[locale] = {"name": name, "description": description}

        if not localizations:
            raise HTTPException(
                status_code=400, 
                detail=f"{index} 행: 최소 하나의 언어에 대한 이름과 설명이 필요합니다."
            )

        rows.append({
            "row": index,
            "product_id": product_id,
            "reference_name": reference_name,
            "type": iap_type,
            "price_tier": price_tier,
            "localizations": localizations,
        })

    if not rows:
        raise HTTPException(status_code=400, detail="등록할 항목이 없습니다.")

    return rows, locales


def _parse_bulk_create_csv(content: bytes) -> tuple[List[Dict[str, Any]], List[str]]:
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="CSV 파일은 UTF-8 인코딩이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV 헤더를 찾을 수 없습니다.")

    languages = sorted({name.split("title_", 1)[1] for name in reader.fieldnames if name.startswith("title_")})
    rows: List[Dict[str, Any]] = []
    seen_skus: set[str] = set()

    for index, row in enumerate(reader, start=2):
        sku = (row.get("sku") or "").strip()
        default_language = (row.get("default_language") or "").strip()
        price_won = (row.get("price_won") or "").strip()
        status = (row.get("status") or "").strip() or None
        row_index = (row.get("index") or "").strip() or None

        if not sku and not default_language and not price_won:
            continue

        if not sku:
            raise HTTPException(status_code=400, detail=f"{index} 행: sku 값을 입력해주세요.")
        if sku in seen_skus:
            raise HTTPException(status_code=400, detail=f"CSV에 중복된 SKU가 있습니다: {sku}")
        seen_skus.add(sku)

        listings: Dict[str, Dict[str, str]] = {}
        for language in languages:
            title_key = f"title_{language}"
            description_key = f"description_{language}"
            title = (row.get(title_key) or "").strip()
            description = (row.get(description_key) or "").strip()
            if title or description:
                listings[language] = {"title": title, "description": description}

        rows.append(
            {
                "row": index,
                "index": row_index,
                "sku": sku,
                "status": status,
                "default_language": default_language,
                "price_won": price_won,
                "listings": listings,
            }
        )

    if not rows:
        raise HTTPException(status_code=400, detail="등록할 항목이 없습니다.")

    return rows, languages


def _build_bulk_create_operations(
    rows: List[Dict[str, Any]],
    existing_products: Dict[str, Dict[str, Any]],
    templates_by_price_micros: Dict[str, PriceTemplate] | None = None,
) -> Dict[str, Any]:
    templates_by_price_micros = templates_by_price_micros or {}
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0}

    existing_skus = {sku for sku, data in existing_products.items() if sku}

    for entry in rows:
        sku = entry["sku"]
        if sku in existing_skus:
            raise HTTPException(status_code=400, detail=f"{sku} 는 이미 등록된 상품입니다.")

        default_language = entry["default_language"] or ""
        if not default_language:
            raise HTTPException(status_code=400, detail=f"{sku} 행: default_language 값을 입력해주세요.")

        price_won_value = entry["price_won"] or ""
        if not price_won_value:
            raise HTTPException(status_code=400, detail=f"{sku} 행: price_won 값을 입력해주세요.")

        try:
            price_micros, _ = _normalize_price_won(price_won_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc

        listings = entry["listings"] or {}
        default_listing = listings.get(default_language)
        if not default_listing or not default_listing.get("title") or not default_listing.get("description"):
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 번역 제목과 설명이 필요합니다.")

        template = templates_by_price_micros.get(price_micros)
        regional_prices = None
        if template:
            pricing_payload = template.to_pricing_payload()
            regional_prices = pricing_payload.get("prices")

        new_product = {
            "sku": sku,
            "status": entry.get("status") or "active",
            "default_language": default_language,
            "default_price": {
                "priceMicros": price_micros,
                "currency": "KRW",
            },
            "listings": listings,
            "prices": regional_prices,
        }

        operations.append({"action": "create", "sku": sku, "data": new_product, "index": entry.get("index")})
        summary["create"] += 1

    return {"operations": operations, "summary": summary}


def _build_import_operations(
    rows: List[Dict[str, Any]],
    existing_products: Dict[str, Dict[str, Any]],
    templates_by_price_micros: Dict[str, PriceTemplate] | None = None,
) -> Dict[str, Any]:
    templates_by_price_micros = templates_by_price_micros or {}
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0, "update": 0, "delete": 0}

    remaining_products = {sku: data for sku, data in existing_products.items()}

    for entry in rows:
        sku = entry["sku"]
        current = remaining_products.pop(sku, None)

        if current is None:
            default_language = entry["default_language"] or ""
            if not default_language:
                raise HTTPException(status_code=400, detail=f"{sku} 행: default_language 값을 입력해주세요.")

            price_won_value = entry["price_won"]
            if not price_won_value:
                logger.info(
                    "Skipping Google Play CSV row for '%s' because price is empty and product is not currently registered.",
                    sku,
                )
                continue
            try:
                price_micros, _ = _normalize_price_won(price_won_value)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc

            currency = "KRW"
            listings = entry["listings"]
            default_listing = listings.get(default_language)
            if not default_listing or not default_listing.get("title") or not default_listing.get("description"):
                raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 번역 제목과 설명이 필요합니다.")

            template = templates_by_price_micros.get(price_micros)
            regional_prices = None
            if template:
                pricing_payload = template.to_pricing_payload()
                regional_prices = pricing_payload.get("prices")

            new_product = {
                "sku": sku,
                "status": entry["status"] or "active",
                "default_language": default_language,
                "default_price": {
                    "priceMicros": price_micros,
                    "currency": currency,
                },
                "listings": listings,
                "prices": regional_prices,
            }
            operations.append({"action": "create", "sku": sku, "data": new_product})
            summary["create"] += 1
            continue

        default_language = entry["default_language"] or current.get("default_language") or ""
        if not default_language:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 정보를 확인할 수 없습니다.")

        status = entry["status"] or current.get("status") or "active"

        current_default_price = current.get("default_price") or {}
        current_price_micros = str(current_default_price.get("priceMicros") or "")
        current_currency = (current_default_price.get("currency") or "KRW").upper()

        applied_template = False
        template_prices: Optional[Dict[str, Any]] = None
        price_micros = current_price_micros
        currency = current_currency or "KRW"

        if entry["price_won"]:
            try:
                normalized_price_micros, _ = _normalize_price_won(entry["price_won"])
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"{sku} 행: {exc}") from exc
            price_micros = normalized_price_micros
            currency = "KRW"

            same_default_price = (
                current_price_micros
                and normalized_price_micros == current_price_micros
                and current_currency == "KRW"
            )

            if not same_default_price:
                template = templates_by_price_micros.get(normalized_price_micros)
            else:
                template = None

            if template:
                pricing_payload = template.to_pricing_payload()
                template_prices = pricing_payload.get("prices")
                applied_template = True
        else:
            price_micros = current_price_micros
            currency = current_currency or "KRW"

        if not price_micros or not currency:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 가격 정보를 확인할 수 없습니다.")

        new_listings = {lang: data.copy() for lang, data in (current.get("listings") or {}).items()}
        for language, listing in entry["listings"].items():
            if listing.get("title") and listing.get("description"):
                new_listings[language] = listing

        if default_language not in new_listings:
            raise HTTPException(status_code=400, detail=f"{sku} 행: 기본 언어 번역 정보를 제공해야 합니다.")

        if applied_template:
            new_prices = template_prices
        else:
            new_prices = {
                region: data.copy()
                for region, data in (current.get("prices") or {}).items()
                if isinstance(data, dict)
            }
            for region, price_payload in (entry.get("prices") or {}).items():
                new_prices[region] = price_payload.copy()
            if not new_prices:
                new_prices = None

        new_product = {
            "sku": sku,
            "status": status,
            "default_language": default_language,
            "default_price": {
                "priceMicros": price_micros,
                "currency": currency,
            },
            "listings": new_listings,
            "prices": new_prices,
        }

        changes = _compute_changes(current, new_product)
        if changes:
            operations.append(
                {
                    "action": "update",
                    "sku": sku,
                    "data": new_product,
                    "current": current,
                    "changes": changes,
                }
            )
            summary["update"] += 1

    for remaining in remaining_products.values():
        operations.append({"action": "delete", "sku": remaining.get("sku"), "current": remaining})
        summary["delete"] += 1

    return {"operations": operations, "summary": summary}


def _apple_row_to_payload(
    row: Dict[str, Any], locales: Iterable[str]
) -> AppleImportPayload:
    try:
        cleared_for_sale = _parse_bool_cell(row.get("cleared_for_sale", ""), default=True)
        family_sharable = _parse_bool_cell(row.get("family_sharable", ""), default=False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"행 {row.get('__row')}: {exc}") from exc

    base_territory = row.get("base_territory") or "KOR"
    localizations: List[AppleLocalization] = []
    screenshots_map = row.get("__screenshots") or {}
    for locale in locales:
        name = row.get(f"name_{locale}") or ""
        description = row.get(f"description_{locale}") or ""
        if not name and not description:
            continue
        screenshot_payload = None
        raw_screenshot = screenshots_map.get(locale)
        if raw_screenshot:
            try:
                screenshot_payload = AppleReviewScreenshot(**raw_screenshot)
            except ValidationError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row.get('__row')}: {locale} 로케일의 스크린샷 정보를 확인할 수 없습니다.",
                ) from exc
        localizations.append(
            AppleLocalization(
                locale=locale,
                name=name or "",
                description=description or "",
                review_screenshot=screenshot_payload,
            )
        )

    # Parse price_krw
    price_krw = None
    price_krw_str = (row.get("price_krw") or "").strip()
    if price_krw_str:
        try:
            price_krw = int(price_krw_str)
            if price_krw < 0:
                raise ValueError("price_krw must be non-negative")
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=400,
                detail=f"행 {row.get('__row')}: price_krw 값이 올바르지 않습니다: {price_krw_str}",
            )
    
    # Get product_id (support both productId and product_id)
    product_id = (row.get("productId") or row.get("product_id") or "").strip()
    if not product_id:
        raise HTTPException(
            status_code=400,
            detail=f"행 {row.get('__row')}: productId가 필요합니다.",
        )
    
    # Get type (required for creation)
    purchase_type = (row.get("type") or row.get("purchase_type") or "").strip() or None
    
    # Generate reference_name from product_id if not provided
    reference_name = (row.get("reference_name") or row.get("referenceName") or product_id).strip()
    
    payload = AppleImportPayload(
        product_id=product_id,
        reference_name=reference_name,
        purchase_type=purchase_type,
        cleared_for_sale=cleared_for_sale,
        family_sharable=family_sharable,
        review_note=(row.get("review_note") or "").strip() or None,
        price_krw=price_krw,
        base_territory=base_territory,
        localizations=localizations,
    )
    return payload


def _build_apple_import_operations(
    rows: List[Dict[str, str]], existing_products: Dict[str, Dict[str, Any]], locales: List[str]
) -> Dict[str, Any]:
    operations: List[Dict[str, Any]] = []
    summary = {"create": 0, "update": 0, "delete": 0}

    for row in rows:
        row_number = row.get("__row", "?")
        product_id = (row.get("product_id") or "").strip()
        if not product_id:
            raise HTTPException(status_code=400, detail=f"행 {row_number}: product_id가 필요합니다.")

        action_text = (row.get("action") or "").strip().lower()
        existing = existing_products.get(product_id)

        if action_text == "delete":
            operations.append(
                {
                    "action": "delete",
                    "product_id": product_id,
                    "current": existing,
                }
            )
            summary["delete"] += 1
            continue

        payload = _apple_row_to_payload(row, locales)
        if not payload.reference_name:
            raise HTTPException(
                status_code=400,
                detail=f"행 {row_number}: reference_name 값을 입력해주세요.",
            )
        if not payload.localizations:
            raise HTTPException(
                status_code=400,
                detail=f"행 {row_number}: 최소 1개의 로컬라이제이션이 필요합니다.",
            )

        if action_text == "create" or (action_text != "update" and product_id not in existing_products):
            if payload.purchase_type is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row_number}: create 작업에는 purchase_type이 필요합니다.",
                )
            operations.append(
                {
                    "action": "create",
                    "product_id": product_id,
                    "data": payload.model_dump(),
                }
            )
            summary["create"] += 1
        else:
            if existing is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"행 {row_number}: 업데이트 대상 상품을 찾을 수 없습니다.",
                )
            operations.append(
                {
                    "action": "update",
                    "product_id": product_id,
                    "data": payload.model_dump(),
                    "current": existing,
                }
            )
            summary["update"] += 1

    return {"operations": operations, "summary": summary}

@app.get("/api/google/inapp/list")
async def api_list_inapp(
    token: str | None = Query(default=None),
    refresh: bool = Query(default=False),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    try:
        if refresh:
            refresh_products_from_remote(GOOGLE_STORE, _fetch_google_products)

        cached_products = get_cached_products_only(GOOGLE_STORE)
        if not cached_products:
            if refresh:
                logger.info("Google Play cache is empty after refresh; returning empty list.")
            else:
                logger.debug("Google Play cache empty; returning empty list without remote fetch.")
            return {"items": [], "nextPageToken": None}

        items, next_token = _paginate_cached_products(cached_products, token, page_size)
        return {"items": items, "nextPageToken": next_token}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/google/inapp/export")
@csv_processing_endpoint
async def api_export_inapp() -> StreamingResponse:
    try:
        products = refresh_products_from_remote(GOOGLE_STORE, _fetch_google_products)
        languages = _collect_languages_from_products(products)
        output = io.StringIO()
        fieldnames = ["sku", "status", "default_language", "price_won", "purchase_option_id"]
        for language in languages:
            fieldnames.append(f"title_{language}")
            fieldnames.append(f"description_{language}")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for item in products:
            canonical = _canonicalize_google_product(item)
            if not canonical:
                continue
            default_price = canonical.get("default_price") or {}
            default_price_micros = default_price.get("priceMicros") or ""
            default_currency = default_price.get("currency") or ""
            row = {
                "sku": canonical.get("sku", ""),
                "status": canonical.get("status", ""),
                "default_language": canonical.get("default_language", ""),
                "price_won": _format_price_won_from_micros(default_price_micros, default_currency)
                if default_price_micros
                else "",
                "purchase_option_id": canonical.get("purchase_option_id") or "",
            }
            listings = canonical.get("listings") or {}
            for language in languages:
                listing = listings.get(language) or {}
                row[f"title_{language}"] = listing.get("title", "")
                row[f"description_{language}"] = listing.get("description", "")
            writer.writerow(row)
        csv_content = output.getvalue()
        csv_bytes = csv_content.encode("utf-8-sig")
        headers = {"Content-Disposition": 'attachment; filename="iap-products.csv"'}
        return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers=headers)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to export in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/google/inapp/new/template")
@csv_processing_endpoint
async def api_bulk_create_template(
    row_count: int = Query(default=20, ge=1, le=500, description="템플릿에 포함할 행 수"),
) -> StreamingResponse:
    try:
        products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        languages = _collect_languages_from_products(products)
    except Exception as exc:
        logger.exception("Failed to prepare languages for bulk template", exc_info=exc)
        languages = []

    if not languages:
        languages = ["ko-KR"]

    fieldnames = ["index", "sku", "status", "default_language", "price_won", "purchase_option_id"]
    for language in languages:
        fieldnames.append(f"title_{language}")
        fieldnames.append(f"description_{language}")

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for idx in range(1, row_count + 1):
        writer.writerow({"index": idx})

    csv_bytes = output.getvalue().encode("utf-8-sig")
    headers = {"Content-Disposition": 'attachment; filename="iap-new-products-template.csv"'}
    return StreamingResponse(iter([csv_bytes]), media_type="text/csv", headers=headers)


@app.get("/api/google/pricing/templates")
async def api_list_price_templates():
    try:
        products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        templates = generate_price_templates_from_products(products)
        return {"templates": [template.to_response() for template in templates]}
    except Exception as exc:
        logger.exception("Failed to load price templates")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/new/import/preview")
@csv_processing_endpoint
async def api_bulk_create_preview(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_languages = _parse_bulk_create_csv(content)

    try:
        existing_products_raw = refresh_products_from_remote(
            GOOGLE_STORE, _fetch_google_products
        )
        existing_products: Dict[str, Dict[str, Any]] = {}
        for raw in existing_products_raw:
            canonical = _canonicalize_google_product(raw)
            if not canonical:
                continue
            sku = canonical.get("sku")
            if sku:
                existing_products[sku] = canonical
        templates = generate_price_templates_from_products(existing_products_raw)
        templates_by_price = index_templates_by_price_micros(templates)
    except Exception as exc:
        logger.exception("Failed to prepare bulk create preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    operations_payload = _build_bulk_create_operations(rows, existing_products, templates_by_price)

    languages = sorted(set(csv_languages))

    return {
        "languages": languages,
        "operations": operations_payload["operations"],
        "summary": operations_payload["summary"],
    }


@app.post("/api/google/inapp/create")
async def api_create_inapp(payload: CreateInAppRequest):
    try:
        allowed_regions: Optional[Set[str]] = None
        try:
            cached_products_only = get_cached_products_only(GOOGLE_STORE)
            allowed = _collect_billable_regions_from_products(cached_products_only)
            if allowed:
                allowed_regions = allowed
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("Failed to collect billable regions from cache: %s", exc)

        regional_pricing = None
        if payload.price_template_id:
            products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
            templates = generate_price_templates_from_products(products)
            template = get_template_by_id(templates, payload.price_template_id)
            if not template:
                raise HTTPException(status_code=400, detail="유효하지 않은 가격 템플릿입니다.")
            regional_pricing = template.to_pricing_payload()
        created = create_onetime_product(
            sku=payload.sku,
            default_language=payload.default_language,
            price_won=payload.price_won,
            regional_pricing=regional_pricing,
            translations=[t.dict() for t in payload.translations],
            allowed_regions=allowed_regions,
        )
        upsert_cached_product(GOOGLE_STORE, created)
        return created
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/new/import/apply")
@csv_processing_endpoint
async def api_bulk_create_apply(request: BulkCreateApplyRequest):
    results = {"create": 0, "failed": 0}
    failed_items = []

    try:
        existing_products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        existing_skus = {item.get("sku") for item in existing_products if item.get("sku")}
        initial_allowed_regions = _collect_billable_regions_from_products(existing_products)
        session_allowed_regions: Optional[Set[str]] = (
            set(initial_allowed_regions) if initial_allowed_regions else None
        )

        for idx, op in enumerate(request.operations, start=1):
            try:
                data = op.data
                if data.sku in existing_skus:
                    raise HTTPException(status_code=400, detail=f"{data.sku} 는 이미 등록된 상품입니다.")

                default_price = data.default_price
                try:
                    price_micros = int(default_price.priceMicros)
                except (TypeError, ValueError):
                    raise HTTPException(
                        status_code=400,
                        detail=f"{data.sku} 행: priceMicros 값이 올바르지 않습니다.",
                    )
                currency = (default_price.currency or "").upper()
                if currency != "KRW":
                    raise HTTPException(
                        status_code=400,
                        detail=f"{data.sku} 행: 기본 통화는 KRW여야 합니다. (현재: {currency or '미지정'})",
                    )
                price_won = price_micros // 1_000_000
                if price_won <= 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{data.sku} 행: priceMicros 값이 유효하지 않습니다.",
                    )

                translations = [
                    {"language": language, "title": listing.title, "description": listing.description}
                    for language, listing in data.listings.items()
                    if listing.title and listing.description
                ]

                regional_pricing: Optional[Dict[str, Any]] = None
                if data.prices:
                    converted_prices: Dict[str, Dict[str, str]] = {}
                    for region, price in data.prices.items():
                        if not isinstance(price, PricePayload):
                            continue
                        if not price.priceMicros or not price.currency:
                            continue
                        converted_prices[region] = {
                            "priceMicros": price.priceMicros,
                            "currency": price.currency,
                        }
                    if converted_prices:
                        regional_pricing = {"prices": converted_prices}

                created = create_onetime_product(
                    sku=data.sku,
                    default_language=data.default_language,
                    price_won=price_won,
                    regional_pricing=regional_pricing,
                    translations=translations,
                    allowed_regions=session_allowed_regions,
                )
                upsert_cached_product(GOOGLE_STORE, created)
                existing_skus.add(data.sku)
                created_regions = _extract_billable_regions_from_product(created)
                if created_regions:
                    session_allowed_regions = created_regions
                results["create"] += 1
            except HTTPException:
                # Re-raise HTTP exceptions (like duplicate SKU)
                raise
            except Exception as exc:
                # Skip failed items and continue
                error_msg = str(exc)
                logger.error(f"Failed to process row {idx} (SKU: {op.data.sku if hasattr(op, 'data') else 'unknown'}): {error_msg}")
                failed_items.append({
                    "row": idx,
                    "sku": op.data.sku if hasattr(op, 'data') else 'unknown',
                    "action": "create",
                    "error": error_msg
                })
                results["failed"] += 1
                # Continue processing next item
                continue
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to apply bulk create operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_data = {"status": "ok", "summary": results}
    
    if failed_items:
        response_data["failed_items"] = failed_items
        logger.info(f"Bulk create completed with {results['failed']} failures. Successful: {results['create']} creates")
    else:
        logger.info(f"Bulk create completed successfully: {results['create']} creates")

    return response_data


@app.put("/api/google/inapp/{sku}")
async def api_update_inapp(sku: str, payload: UpdateInAppRequest):
    try:
        listings_payload = {
            language: listing.model_dump()
            for language, listing in payload.listings.items()
        }
        prices_payload = (
            {region: price.model_dump() for region, price in payload.prices.items()}
            if payload.prices
            else None
        )
        updated = update_managed_inapp(
            sku=sku,
            default_language=payload.default_language,
            status=payload.status,
            default_price=payload.default_price.model_dump(),
            listings=listings_payload,
            prices=prices_payload,
        )
        upsert_cached_product(GOOGLE_STORE, updated)
        return {"status": "ok"}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to update managed in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/google/inapp/{sku}")
async def api_delete_inapp(sku: str):
    try:
        delete_inapp_product(sku=sku)
        delete_cached_product(GOOGLE_STORE, sku)
        return {"status": "ok"}
    except Exception as exc:
        logger.exception("Failed to delete in-app product")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/google/inapp/import/preview")
@csv_processing_endpoint
async def api_import_preview(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_languages = _parse_import_csv(content)

    try:
        existing_products_raw = refresh_products_from_remote(
            GOOGLE_STORE, _fetch_google_products
        )
        existing_products: Dict[str, Dict[str, Any]] = {}
        for raw in existing_products_raw:
            canonical = _canonicalize_google_product(raw)
            if not canonical:
                continue
            sku = canonical.get("sku")
            if sku:
                existing_products[sku] = canonical
        templates = generate_price_templates_from_products(existing_products_raw)
        templates_by_price = index_templates_by_price_micros(templates)
    except Exception as exc:
        logger.exception("Failed to load existing in-app products for import preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    operations_payload = _build_import_operations(rows, existing_products, templates_by_price)
    existing_languages = {
        lang
        for product in existing_products.values()
        for lang in (product.get("listings") or {}).keys()
    }
    languages = sorted(set(csv_languages) | existing_languages)

    return {
        "languages": languages,
        "operations": operations_payload["operations"],
        "summary": operations_payload["summary"],
    }


@app.post("/api/google/inapp/import/apply")
@csv_processing_endpoint
async def api_import_apply(request: ImportApplyRequest):
    results = {"create": 0, "update": 0, "delete": 0, "failed": 0}
    failed_items = []

    try:
        cached_products = get_cached_products(GOOGLE_STORE, _fetch_google_products)
        initial_allowed_regions = _collect_billable_regions_from_products(cached_products)
        session_allowed_regions: Optional[Set[str]] = (
            set(initial_allowed_regions) if initial_allowed_regions else None
        )

        for idx, op in enumerate(request.operations, start=1):
            try:
                if op.action == "create":
                    if not op.data:
                        raise HTTPException(status_code=400, detail="create 작업에는 data가 필요합니다.")
                    data = op.data

                    default_price = data.default_price
                    try:
                        price_micros = int(default_price.priceMicros)
                    except (TypeError, ValueError):
                        raise HTTPException(
                            status_code=400,
                            detail=f"{data.sku} 행: priceMicros 값이 올바르지 않습니다.",
                        )
                    currency = (default_price.currency or "").upper()
                    if currency != "KRW":
                        raise HTTPException(
                            status_code=400,
                            detail=f"{data.sku} 행: 기본 통화는 KRW여야 합니다. (현재: {currency or '미지정'})",
                        )
                    price_won = price_micros // 1_000_000
                    if price_won <= 0:
                        raise HTTPException(
                            status_code=400,
                            detail=f"{data.sku} 행: priceMicros 값이 유효하지 않습니다.",
                        )

                    translations = [
                        {"language": language, "title": listing.title, "description": listing.description}
                        for language, listing in data.listings.items()
                        if listing.title and listing.description
                    ]

                    regional_pricing: Optional[Dict[str, Any]] = None
                    if data.prices:
                        converted_prices: Dict[str, Dict[str, str]] = {}
                        for region, price in data.prices.items():
                            if not isinstance(price, PricePayload):
                                continue
                            if not price.priceMicros or not price.currency:
                                continue
                            converted_prices[region] = {
                                "priceMicros": price.priceMicros,
                                "currency": price.currency,
                            }
                        if converted_prices:
                            regional_pricing = {"prices": converted_prices}

                    created = create_onetime_product(
                        sku=data.sku,
                        default_language=data.default_language,
                        price_won=price_won,
                        regional_pricing=regional_pricing,
                        translations=translations,
                        allowed_regions=session_allowed_regions,
                    )
                    upsert_cached_product(GOOGLE_STORE, created)
                    created_regions = _extract_billable_regions_from_product(created)
                    if created_regions:
                        session_allowed_regions = created_regions
                    results["create"] += 1
                elif op.action == "update":
                    if not op.data:
                        raise HTTPException(status_code=400, detail="update 작업에는 data가 필요합니다.")
                    data = op.data
                    listings_payload = {
                        language: listing.model_dump()
                        for language, listing in data.listings.items()
                    }
                    updated = update_managed_inapp(
                        sku=data.sku,
                        default_language=data.default_language,
                        status=data.status,
                        default_price=data.default_price.model_dump(),
                        listings=listings_payload,
                        prices=data.prices,
                    )
                    upsert_cached_product(GOOGLE_STORE, updated)
                    results["update"] += 1
                elif op.action == "delete":
                    delete_inapp_product(sku=op.sku)
                    delete_cached_product(GOOGLE_STORE, op.sku)
                    results["delete"] += 1
                else:  # pragma: no cover - defensive
                    raise HTTPException(status_code=400, detail=f"지원하지 않는 작업 유형입니다: {op.action}")
            except HTTPException:
                # Re-raise HTTP exceptions (like validation errors)
                raise
            except Exception as exc:
                # Skip failed items and continue
                error_msg = str(exc)
                logger.error(f"Failed to process row {idx} (SKU: {getattr(op, 'sku', 'unknown')}): {error_msg}")
                failed_items.append({
                    "row": idx,
                    "sku": getattr(op, 'sku', op.data.sku if hasattr(op, 'data') and hasattr(op.data, 'sku') else 'unknown'),
                    "action": op.action,
                    "error": error_msg
                })
                results["failed"] += 1
                # Continue processing next item
                continue
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except Exception as exc:
        logger.exception("Failed to apply import operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_data = {"status": "ok", "summary": results}
    
    if failed_items:
        response_data["failed_items"] = failed_items
        logger.info(f"Import completed with {results['failed']} failures. Successful: {results['create']} creates, {results['update']} updates, {results['delete']} deletes")
    else:
        logger.info(f"Import completed successfully: {results['create']} creates, {results['update']} updates, {results['delete']} deletes")

    return response_data


@app.get("/api/apple/config")
async def api_apple_config():
    return {
        "localization": {
            "defaultLocale": APPLE_DEFAULT_LOCALIZATION_LOCALE,
            "locales": list(APPLE_LOCALIZATION_LOCALES),
        },
        "pricing": {
            "fixedPriceTerritories": list(get_fixed_price_territories()),
        },
    }


@app.get("/api/apple/inapp/bulk-create/sample")
@csv_processing_endpoint
async def api_apple_bulk_create_sample() -> StreamingResponse:
    """Generate sample CSV for Apple IAP bulk creation.
    
    Returns a CSV with 2 rows based on an existing IAP (if available) with full localization data.
    """
    try:
        # Get existing products
        products = await _run_in_thread(get_cached_products, APPLE_STORE, _fetch_apple_products, False)
        
        # Build CSV header
        locales = APPLE_LOCALIZATION_LOCALES
        fields = ['productId', 'type', 'price_krw']
        for locale in locales:
            fields.append(f'name_{locale}')
            fields.append(f'description_{locale}')
        
        rows = [','.join(fields)]
        
        # Get first product as sample (fetch full detail with localizations)
        sample_product = None
        if products:
            first_product = products[0]
            product_id = first_product.get('productId') or first_product.get('sku')
            if product_id:
                try:
                    # Fetch full detail with localizations
                    detail = await _run_in_thread(_find_apple_inapp, product_id)
                    if detail:
                        sample_product = detail
                except Exception as exc:
                    logger.warning("Failed to fetch sample product detail: %s", exc)
        
        # Generate 2 sample rows
        for row_num in range(2):
            if sample_product:
                # Use different product_id for each row (add suffix)
                base_product_id = str(sample_product.get('productId') or sample_product.get('sku') or 'sample_product')
                product_id = f'{base_product_id}_sample_{row_num + 1}'
                product_type = str(sample_product.get('type') or 'consumable')
                
                # Get KRW price
                price_krw = ''
                krw_price = sample_product.get('krwPrice')
                if isinstance(krw_price, dict) and krw_price.get('customerPrice'):
                    try:
                        price_str = str(krw_price.get('customerPrice', ''))
                        if price_str:
                            price_krw = str(int(float(price_str)))
                    except (ValueError, TypeError):
                        price_krw = '1000'  # Default sample price
                else:
                    price_krw = '1000'  # Default sample price
                
                # Get localizations
                localizations = sample_product.get('localizations') or []
                localization_map = {loc.get('locale'): loc for loc in localizations if isinstance(loc, dict)}
            else:
                # No existing product, use default values
                product_id = f'sample_product_{row_num + 1}'
                product_type = 'consumable'
                price_krw = '1000'
                localization_map = {}
            
            row_data = [product_id, product_type, price_krw]
            
            for locale in locales:
                if sample_product and locale in localization_map:
                    loc = localization_map[locale]
                    name = str(loc.get('name') or '').replace('"', '""')
                    description = str(loc.get('description') or '').replace('"', '""')
                else:
                    name = f'Sample Product {row_num + 1}'
                    description = f'Sample Description {row_num + 1}'
                row_data.append(f'"{name}"')
                row_data.append(f'"{description}"')
            
            rows.append(','.join(row_data))
        
        csv_content = '\n'.join(rows).encode('utf-8-sig')  # UTF-8 with BOM
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"apple-iap-bulk-create-sample-{timestamp}.csv"
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception("Failed to generate Apple IAP bulk create sample")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/import/preview")
@csv_processing_endpoint
async def api_apple_import_preview(file: UploadFile = File(...)):
    """Preview Apple IAP bulk creation from CSV (create only, no update/delete)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    try:
        # Parse CSV using existing parser
        rows, locales = await _run_in_thread(_parse_apple_import_csv, content, None)
        
        if not rows:
            raise HTTPException(status_code=400, detail="CSV 파일에 데이터가 없습니다.")
        
        # Get existing products to check for duplicates
        existing_products_list = await _run_in_thread(get_cached_products, APPLE_STORE, _fetch_apple_products, False)
        existing_product_ids = {
            item.get("productId") 
            for item in existing_products_list 
            if item.get("productId")
        }
        
        # Build operations (create only, skip duplicates)
        operations: List[Dict[str, Any]] = []
        summary = {"create": 0, "duplicate": 0, "invalid": 0}
        duplicate_items = []
        invalid_items = []
        
        for row in rows:
            row_number = row.get("__row", "?")
            product_id = (row.get("productId") or row.get("product_id") or "").strip()
            
            if not product_id:
                invalid_items.append({
                    "row": row_number,
                    "product_id": "",
                    "reason": "productId가 필요합니다."
                })
                summary["invalid"] += 1
                continue
            
            # Check for duplicates
            if product_id in existing_product_ids:
                duplicate_items.append({
                    "row": row_number,
                    "product_id": product_id,
                })
                summary["duplicate"] += 1
                continue
            
            try:
                payload = _apple_row_to_payload(row, locales)
                if not payload.reference_name:
                    invalid_items.append({
                        "row": row_number,
                        "product_id": product_id,
                        "reason": "reference_name이 필요합니다."
                    })
                    summary["invalid"] += 1
                    continue
                if not payload.localizations:
                    invalid_items.append({
                        "row": row_number,
                        "product_id": product_id,
                        "reason": "최소 1개의 로컬라이제이션이 필요합니다."
                    })
                    summary["invalid"] += 1
                    continue
                if payload.purchase_type is None:
                    invalid_items.append({
                        "row": row_number,
                        "product_id": product_id,
                        "reason": "type이 필요합니다."
                    })
                    summary["invalid"] += 1
                    continue
                
                operations.append({
                    "action": "create",
                    "product_id": product_id,
                    "data": payload.model_dump(),
                    "row": row_number,
                })
                summary["create"] += 1
            except HTTPException as exc:
                invalid_items.append({
                    "row": row_number,
                    "product_id": product_id,
                    "reason": str(exc.detail) if hasattr(exc, 'detail') else str(exc),
                })
                summary["invalid"] += 1
                continue
        
        return {
            "locales": locales,
            "operations": operations,
            "summary": summary,
            "duplicate_items": duplicate_items,
            "invalid_items": invalid_items,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to prepare Apple import preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/import/apply")
@csv_processing_endpoint
async def api_apple_import_apply(request: Request):
    """Apply Apple IAP bulk creation from CSV (create only, no update/delete)."""
    results = {"create": 0, "failed": 0, "duplicate": 0}
    success_items = []
    failed_items = []
    duplicate_items = []
    
    try:
        payload = await request.json()
        operations = payload.get("operations", [])
        
        # Get existing products to check for duplicates
        existing_products_list = await _run_in_thread(get_cached_products, APPLE_STORE, _fetch_apple_products, False)
        existing_product_ids = {
            item.get("productId") 
            for item in existing_products_list 
            if item.get("productId")
        }
        
        for idx, op in enumerate(operations, start=1):
            try:
                action = op.get("action")
                product_id = op.get("product_id")
                data = op.get("data", {})
                
                # Only allow create action
                if action != "create":
                    failed_items.append({
                        "row": op.get("row", idx),
                        "product_id": product_id,
                        "action": action,
                        "error": f"지원하지 않는 작업입니다: {action} (신규 등록만 가능합니다)"
                    })
                    results["failed"] += 1
                    continue
                
                # Check for duplicates
                if product_id in existing_product_ids:
                    duplicate_items.append({
                        "row": op.get("row", idx),
                        "product_id": product_id,
                    })
                    results["duplicate"] += 1
                    continue
                
                # Create new IAP
                def _create_apple_iap() -> Dict[str, Any]:
                    payload_data = AppleImportPayload(**data)
                    result = create_apple_inapp_purchase(
                        product_id=product_id,
                        reference_name=payload_data.reference_name,
                        purchase_type=payload_data.purchase_type or "consumable",
                        cleared_for_sale=payload_data.cleared_for_sale,
                        family_sharable=payload_data.family_sharable,
                        review_note=payload_data.review_note,
                        price_point_id=None,
                        price_tier=None,
                        price_krw=payload_data.price_krw,
                        base_territory=payload_data.base_territory,
                        localizations=[loc.model_dump() for loc in payload_data.localizations],
                    )
                    upsert_cached_product(APPLE_STORE, _to_apple_summary(result))
                    return result
                
                await _run_in_thread(_create_apple_iap)
                results["create"] += 1
                success_items.append({
                    "row": op.get("row", idx),
                    "product_id": product_id,
                })
                # Add to existing set to prevent duplicate creation in same batch
                existing_product_ids.add(product_id)
                
            except HTTPException:
                raise
            except Exception as exc:
                failed_items.append({
                    "row": op.get("row", idx),
                    "product_id": op.get("product_id", "unknown"),
                    "action": op.get("action", "unknown"),
                    "error": str(exc)
                })
                results["failed"] += 1
                logger.error(f"Failed to process row {op.get('row', idx)}: {exc}")
                continue
        
        response_data = {"status": "ok", "summary": results}
        if success_items:
            response_data["success_items"] = success_items
        if failed_items:
            response_data["failed_items"] = failed_items
        if duplicate_items:
            response_data["duplicate_items"] = duplicate_items
        
        return response_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to apply Apple import operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/inapp/export")
@csv_processing_endpoint
async def api_apple_export() -> StreamingResponse:
    """Export all Apple IAPs to CSV.
    
    CSV format:
    - productId: 상품 ID
    - type: 타입 (consumable, nonConsumable, etc.)
    - price_krw: KRW 가격 (정수)
    - name_{locale}: 각 로케일별 제목 (APPLE_LOCALIZATION_LOCALES 기준)
    - description_{locale}: 각 로케일별 설명 (APPLE_LOCALIZATION_LOCALES 기준)
    """
    try:
        # Get all cached IAPs with full relationships
        products = await _run_in_thread(get_cached_products, APPLE_STORE, _fetch_apple_products, False)
        
        # Build CSV header
        locales = APPLE_LOCALIZATION_LOCALES
        fields = ['productId', 'type', 'price_krw']
        for locale in locales:
            fields.append(f'name_{locale}')
            fields.append(f'description_{locale}')
        
        rows = [','.join(fields)]
        
        if not products:
            # Return empty CSV with headers
            csv_content = '\n'.join(rows).encode('utf-8-sig')  # UTF-8 with BOM
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"apple-iap-export-{timestamp}.csv"
            return StreamingResponse(
                iter([csv_content]),
                media_type="text/csv",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        
        # Fetch full details for each product to get localizations
        for product in products:
            product_id = str(product.get('productId') or product.get('sku') or '')
            if not product_id:
                continue
            
            product_type = str(product.get('type') or '')
            
            # Get KRW price
            price_krw = ''
            krw_price = product.get('krwPrice')
            if isinstance(krw_price, dict) and krw_price.get('customerPrice'):
                try:
                    # customerPrice는 문자열일 수 있으므로 정수로 변환
                    price_str = str(krw_price.get('customerPrice', ''))
                    if price_str:
                        # 소수점이 있으면 제거하고 정수로 변환
                        price_krw = str(int(float(price_str)))
                except (ValueError, TypeError):
                    pass
            
            # Build row data
            row_data = [
                product_id,
                product_type,
                price_krw,
            ]
            
            # Add localizations for each locale
            # Try to get localizations from cache or fetch detail if needed
            localizations = product.get('localizations') or []
            localization_map = {loc.get('locale'): loc for loc in localizations if isinstance(loc, dict)}
            
            for locale in locales:
                loc = localization_map.get(locale, {})
                name = str(loc.get('name') or '').replace('"', '""')
                description = str(loc.get('description') or '').replace('"', '""')
                row_data.append(f'"{name}"' if name else '')
                row_data.append(f'"{description}"' if description else '')
            
            rows.append(','.join(row_data))
        
        csv_content = '\n'.join(rows).encode('utf-8-sig')  # UTF-8 with BOM
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"apple-iap-export-{timestamp}.csv"
        
        return StreamingResponse(
            iter([csv_content]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as exc:
        logger.exception("Failed to export Apple in-app products")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/inapp/list")
async def api_apple_list_inapp(
    token: str | None = Query(default=None),
    refresh: bool = Query(default=False),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=200),
):
    try:
        if refresh:
            try:
                logger.info("Manual refresh requested - performing full Apple IAP fetch")
                await _run_in_thread(
                    refresh_products_from_remote, APPLE_STORE, _fetch_apple_products
                )
            except AppleStoreConfigError as exc:
                logger.warning(
                    "Failed to refresh Apple products: %s. Continuing with cached data.",
                    exc
                )
                # Continue with cached data instead of failing
        
        # Get total count if it's a fresh request
        total_count = None
        try:
            if not token:
                if refresh:
                    # Use lightweight ID fetch to get total count without fetching full data again
                    _, total_count = await _run_in_thread(get_inapp_purchase_ids_lightweight)
                else:
                    # For non-refresh requests, get count from cache
                    cached_items = await _run_in_thread(
                        get_cached_products, APPLE_STORE, _fetch_apple_products, False
                    )
                    if cached_items:
                        total_count = len(cached_items)
        except Exception:
            pass  # Ignore errors when getting total count
        
        items, next_token = await _run_in_thread(
            get_paginated_products,
            APPLE_STORE,
            _fetch_apple_products,
            token,
            page_size=page_size,
        )
        
        return {
            "items": items, 
            "nextPageToken": next_token,
            "totalCount": total_count
        }
    except HTTPException:
        raise
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to list Apple in-app purchases")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/batch/preview")
@csv_processing_endpoint
async def api_apple_batch_create_preview(file: UploadFile = File(...)):
    """Preview Apple IAP batch creation from CSV (only new IAPs)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="CSV 파일이 비어 있습니다.")

    rows, csv_locales = await _run_in_thread(_parse_apple_batch_create_csv, content)

    try:
        # Fetch existing products
        existing_products_raw, _ = await _run_in_thread(
            get_all_inapp_purchases, include_relationships=False
        )
        existing_product_ids = {
            (item.get("productId") or item.get("sku") or "").strip()
            for item in existing_products_raw
        }
        
        # Validate and build operations
        operations: List[Dict[str, Any]] = []
        errors: List[str] = []
        
        for entry in rows:
            product_id = entry["product_id"]
            
            # Check if product already exists
            if product_id in existing_product_ids:
                errors.append(f"행 {entry['row']}: {product_id}는 이미 등록된 상품입니다.")
                continue
            
            # Validate IAP type
            iap_type = entry["type"].lower()
            valid_types = ["consumable", "non_consumable", "non_renewing_subscription"]
            if iap_type not in valid_types:
                errors.append(
                    f"행 {entry['row']}: type은 {', '.join(valid_types)} 중 하나여야 합니다. (현재: {entry['type']})"
                )
                continue
            
            operations.append({
                "action": "create",
                "product_id": product_id,
                "data": entry
            })
        
        if errors:
            raise HTTPException(status_code=400, detail="\n".join(errors))
        
        summary = {"create": len(operations)}
        
        return {
            "locales": sorted(csv_locales),
            "operations": operations,
            "summary": summary,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to prepare Apple batch create preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/batch/apply")
async def api_apple_batch_create_apply(request: Request):
    """Apply Apple IAP batch creation operations."""
    results = {"create": 0, "failed": 0}
    errors: List[str] = []

    try:
        payload = await request.json()
        operations = payload.get("operations", [])
        
        # Fetch existing products to double-check
        existing_products_raw, _ = await _run_in_thread(
            get_all_inapp_purchases, include_relationships=False
        )
        existing_product_ids = {
            (item.get("productId") or item.get("sku") or "").strip()
            for item in existing_products_raw
        }

        for op in operations:
            if op.get("action") != "create":
                continue
                
            data = op.get("data", {})
            product_id = data.get("product_id", "")
            reference_name = data.get("reference_name", "")
            iap_type = data.get("type", "consumable")
            price_tier = data.get("price_tier")
            localizations_dict = data.get("localizations", {})
            
            # Double-check if product already exists
            if product_id in existing_product_ids:
                errors.append(f"{product_id}는 이미 등록된 상품입니다.")
                results["failed"] += 1
                continue
            
            try:
                # Prepare localizations
                localizations = []
                for locale, texts in localizations_dict.items():
                    localizations.append({
                        "locale": locale,
                        "name": texts.get("name", ""),
                        "description": texts.get("description", ""),
                    })
                
                if not localizations:
                    errors.append(f"{product_id}: 최소 하나의 언어 번역이 필요합니다.")
                    results["failed"] += 1
                    continue
                
                # Map IAP type to Apple's format
                purchase_type_map = {
                    "consumable": "CONSUMABLE",
                    "non_consumable": "NON_CONSUMABLE",
                    "non_renewing_subscription": "NON_RENEWING_SUBSCRIPTION",
                }
                purchase_type = purchase_type_map.get(iap_type.lower(), "CONSUMABLE")
                
                # Create the IAP
                created = await _run_in_thread(
                    create_apple_inapp_purchase,
                    product_id=product_id,
                    reference_name=reference_name,
                    purchase_type=purchase_type,
                    cleared_for_sale=False,  # Default to not cleared for sale
                    family_sharable=False,  # Default to not family sharable
                    review_note=None,
                    price_point_id=data.get("price_point_id"),
                    price_tier=price_tier,
                    price_krw=data.get("price_krw"),
                    base_territory="KOR",  # Default territory
                    localizations=localizations,
                )
                
                await _run_in_thread(upsert_cached_product, APPLE_STORE, created)
                existing_product_ids.add(product_id)
                results["create"] += 1
            except Exception as exc:
                logger.error("Failed to create Apple IAP %s: %s", product_id, exc)
                errors.append(f"{product_id}: {str(exc)}")
                results["failed"] += 1
        
        response = {"status": "ok" if not errors else "partial", "summary": results}
        if errors:
            response["errors"] = errors
        
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to apply Apple batch create operations")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/pricing/tiers")
async def api_apple_price_tiers(territory: str = Query(default="KOR", min_length=2)):
    try:
        tiers = await _run_in_thread(list_apple_price_tiers, territory)
        return {"tiers": tiers}
    except AppleStorePermissionError as exc:
        logger.warning("Apple Store permission error for price tiers: %s", exc)
        # Return empty list instead of error if permissions are missing
        return {"tiers": []}
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load Apple price tiers")
        # Return empty list instead of 500 error
        return {"tiers": []}


@app.post("/api/apple/inapp/create")
async def api_apple_create_inapp(payload: AppleCreateInAppRequest):
    try:
        def _create_and_cache() -> Dict[str, Any]:
            result = create_apple_inapp_purchase(
                product_id=payload.product_id,
                reference_name=payload.reference_name,
                purchase_type=payload.purchase_type,
                cleared_for_sale=payload.cleared_for_sale,
                family_sharable=payload.family_sharable,
                review_note=payload.review_note,
                price_point_id=payload.price_point_id,
                price_tier=None,
                price_krw=payload.price_krw,
                base_territory=payload.base_territory,
                localizations=[loc.model_dump() for loc in payload.localizations],
            )
            upsert_cached_product(APPLE_STORE, _to_apple_summary(result))
            return result

        result = await _run_in_thread(_create_and_cache)
        return {"status": "ok", "item": result}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to create Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _find_apple_inapp(product_id: str) -> Dict[str, Any]:
    normalized = (product_id or "").strip()
    if not normalized:
        raise HTTPException(status_code=404, detail="Apple 인앱 상품을 찾을 수 없습니다.")

    def _lookup(products: Iterable[Dict[str, Any]]) -> Optional[str]:
        for item in products:
            if not isinstance(item, dict):
                continue
            candidate = item.get("productId") or item.get("sku")
            if candidate == normalized:
                inapp_id = item.get("id")
                if isinstance(inapp_id, str) and inapp_id:
                    return inapp_id
        return None

    products = get_cached_products(APPLE_STORE, _fetch_apple_products)
    inapp_id = _lookup(products)
    if not inapp_id:
        products = refresh_products_from_remote(APPLE_STORE, _fetch_apple_products)
        inapp_id = _lookup(products)

    if not inapp_id:
        raise HTTPException(status_code=404, detail="해당 Apple 인앱 상품을 찾을 수 없습니다.")

    try:
        # Fetch detail with localizations (include_relationships=True by default)
        detail = get_apple_inapp_purchase_detail(inapp_id)
        
        # Update cache with summary (including localization info if available)
        # This ensures that when the user edits the IAP, the localization info is cached
        if detail:
            summary = _to_apple_summary(detail)
            upsert_cached_product(APPLE_STORE, summary)
            logger.debug("Updated cache for Apple IAP %s with localization info", normalized)
        
        return detail
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to fetch Apple in-app purchase detail")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/apple/inapp/{product_id}/detail")
async def api_apple_get_inapp_detail(product_id: str, skip_price_details: bool = Query(default=False)):
    """Get Apple IAP detail. If skip_price_details=True, excludes proceeds and priceTier from price info."""
    def _find_with_price_control() -> Dict[str, Any]:
        normalized = (product_id or "").strip()
        if not normalized:
            raise HTTPException(status_code=404, detail="Apple 인앱 상품을 찾을 수 없습니다.")

        def _lookup(products: Iterable[Dict[str, Any]]) -> Optional[str]:
            for item in products:
                if not isinstance(item, dict):
                    continue
                candidate = item.get("productId") or item.get("sku")
                if candidate == normalized:
                    inapp_id = item.get("id")
                    if isinstance(inapp_id, str) and inapp_id:
                        return inapp_id
            return None

        products = get_cached_products(APPLE_STORE, _fetch_apple_products)
        inapp_id = _lookup(products)
        if not inapp_id:
            products = refresh_products_from_remote(APPLE_STORE, _fetch_apple_products)
            inapp_id = _lookup(products)

        if not inapp_id:
            raise HTTPException(status_code=404, detail="해당 Apple 인앱 상품을 찾을 수 없습니다.")

        try:
            # Fetch detail with localizations (include_relationships=True by default)
            detail = get_apple_inapp_purchase_detail(inapp_id)
            
            # If skip_price_details is True, remove proceeds and priceTier from price info
            if skip_price_details and detail:
                krw_price = detail.get("krwPrice")
                if isinstance(krw_price, dict):
                    # Create a copy without proceeds and priceTier
                    filtered_price = {
                        "currency": krw_price.get("currency"),
                        "customerPrice": krw_price.get("customerPrice"),
                    }
                    detail["krwPrice"] = filtered_price
                # Also remove priceTier from top level if present
                if "priceTier" in detail:
                    detail.pop("priceTier")
            
            # Update cache with summary (including localization info if available)
            # This ensures that when the user edits the IAP, the localization info is cached
            if detail:
                summary = _to_apple_summary(detail)
                upsert_cached_product(APPLE_STORE, summary)
                logger.debug("Updated cache for Apple IAP %s with localization info", normalized)
            
            return detail
        except AppleStoreConfigError as exc:
            logger.error("Apple Store configuration error: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Failed to fetch Apple in-app purchase detail")
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    
    item = await _run_in_thread(_find_with_price_control)
    return {"item": item}


@app.put("/api/apple/inapp/{product_id}")
async def api_apple_update_inapp(product_id: str, payload: AppleUpdateInAppRequest):
    try:
        def _update_and_cache() -> Dict[str, Any]:
            target = _find_apple_inapp(product_id)
            inapp_id = target.get("id")
            if not inapp_id:
                raise HTTPException(status_code=404, detail="Apple 인앱 상품 ID를 확인할 수 없습니다.")

            localizations = payload.localizations or [
                AppleLocalization(locale=locale, name=data.get("name", ""), description=data.get("description", ""))
                for locale, data in (target.get("localizations") or {}).items()
            ]

            result = update_apple_inapp_purchase(
                inapp_id=inapp_id,
                reference_name=payload.reference_name,
                cleared_for_sale=payload.cleared_for_sale,
                family_sharable=payload.family_sharable,
                review_note=payload.review_note,
                price_point_id=payload.price_point_id,
                price_tier=None,
                price_krw=payload.price_krw,
                base_territory=payload.base_territory,
                localizations=[loc.model_dump() for loc in localizations],
            )
            upsert_cached_product(APPLE_STORE, _to_apple_summary(result))
            return result

        result = await _run_in_thread(_update_and_cache)
        return {"status": "ok", "item": result}
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to update Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/api/apple/inapp/{product_id}")
async def api_apple_delete_inapp(product_id: str):
    try:
        def _delete_and_purge() -> None:
            target = _find_apple_inapp(product_id)
            inapp_id = target.get("id")
            if not inapp_id:
                raise HTTPException(status_code=404, detail="Apple 인앱 상품 ID를 확인할 수 없습니다.")
            delete_apple_inapp_purchase(inapp_id)
            delete_cached_product(APPLE_STORE, product_id)

        await _run_in_thread(_delete_and_purge)
        return {"status": "ok"}
    except HTTPException:
        raise
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to delete Apple in-app purchase")
        raise HTTPException(status_code=500, detail=str(exc)) from exc




@app.post("/api/apple/inapp/bulk-delete/preview")
async def api_apple_bulk_delete_preview(request: AppleBulkDeletePreviewRequest):
    try:
        products = await _run_in_thread(
            refresh_products_from_remote, APPLE_STORE, _fetch_apple_products
        )
    except AppleStoreConfigError as exc:
        logger.error("Apple Store configuration error: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to load Apple in-app purchases for bulk delete preview")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    lookup_field = {
        "reference_name": "referenceName",
        "product_id": "productId",
        "sku": "productId",
        "iap_id": "id",
    }[request.identifier_type]

    index: Dict[str, List[Dict[str, Any]]] = {}
    for item in products:
        if not isinstance(item, dict):
            continue
        value = item.get(lookup_field)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        index.setdefault(normalized, []).append(item)

    matches: List[Dict[str, Any]] = []
    for raw in request.values:
        normalized = raw.strip()
        for item in index.get(normalized, []):
            summary = _to_apple_summary(item)
            if not summary:
                continue
            summary["matchedValue"] = normalized
            summary.setdefault("identifierType", request.identifier_type)
            matches.append(summary)

    return {"matches": matches}


@app.post("/api/apple/inapp/bulk-delete/apply")
async def api_apple_bulk_delete_apply(request: AppleBulkDeleteApplyRequest):
    try:
        def _apply_bulk_delete() -> Dict[str, Any]:
            summary = {"deleted": 0, "failed": []}

            for entry in request.items:
                try:
                    delete_apple_inapp_purchase(entry.inapp_id)
                except AppleStoreConfigError as exc:
                    logger.error("Apple Store configuration error: %s", exc)
                    raise HTTPException(status_code=503, detail=str(exc)) from exc
                except Exception as exc:
                    logger.exception(
                        "Failed to delete Apple in-app purchase %s", entry.inapp_id
                    )
                    summary["failed"].append(
                        {
                            "inapp_id": entry.inapp_id,
                            "product_id": entry.product_id,
                            "reason": str(exc),
                        }
                    )
                    continue

                delete_cached_product(APPLE_STORE, entry.product_id)
                summary["deleted"] += 1

            status = "ok" if not summary["failed"] else "partial"
            return {"status": status, "summary": summary}

        return await _run_in_thread(_apply_bulk_delete)
    except HTTPException:
        raise


def _extract_product_key(sku: str) -> Optional[str]:
    """Extract product key from SKU (e.g., 'aos_dg_1126_99' -> 'dg_1126').
    
    Google: aos_dg_xxxx_xxxx -> dg_xxxx
    iOS: ios_dg_xxxx_xxxx -> dg_xxxx
    
    Also handles:
    - aos_dg_1098_550 -> dg_1098
    - ios_dg_1098_550 -> dg_1098
    - aos_dg_1098_550f -> dg_1098
    """
    if not sku:
        return None
    
    sku = sku.strip()
    parts = sku.split('_')
    
    # 패턴: [prefix]_dg_[number]_[suffix]
    # prefix는 'aos' 또는 'ios'일 수 있음
    # dg 다음의 숫자 부분을 추출
    if len(parts) >= 3:
        # parts[0]은 'aos' 또는 'ios'
        # parts[1]은 'dg'여야 함
        if parts[1] == 'dg' and len(parts) >= 3:
            # parts[2]가 숫자인지 확인
            try:
                # 숫자 부분 추출 (앞의 0 제거 가능)
                number_part = parts[2].lstrip('0') or '0'
                # 'dg_xxxx' 형식으로 반환
                return f"{parts[1]}_{number_part}"
            except (ValueError, IndexError):
                pass
    
    # 대소문자 구분 없이 재시도
    parts_lower = [p.lower() for p in parts]
    if len(parts_lower) >= 3 and parts_lower[1] == 'dg':
        try:
            number_part = parts_lower[2].lstrip('0') or '0'
            return f"dg_{number_part}"
        except (ValueError, IndexError):
            pass
    
    return None


def _get_protected_apple_ids() -> Set[str]:
    raw = get_metadata_value(APPLE_PROTECTED_KEY)
    if not raw:
        return set()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return {str(item).strip() for item in data if str(item).strip()}
    except json.JSONDecodeError:
        logger.warning("Failed to parse protected Apple IAP metadata; resetting.")
    return set()


def _set_protected_apple_ids(ids: Iterable[str]) -> None:
    cleaned = sorted({str(item).strip() for item in ids if str(item).strip()})
    set_metadata_value(APPLE_PROTECTED_KEY, json.dumps(cleaned, ensure_ascii=False))


def _add_protected_apple_ids(ids: Iterable[str]) -> Set[str]:
    current = _get_protected_apple_ids()
    updated = current.union({str(item).strip() for item in ids if str(item).strip()})
    _set_protected_apple_ids(updated)
    return updated


def _remove_protected_apple_ids(ids: Iterable[str]) -> Set[str]:
    to_remove = {str(item).strip() for item in ids if str(item).strip()}
    current = _get_protected_apple_ids()
    if not current:
        return current
    updated = {item for item in current if item not in to_remove}
    _set_protected_apple_ids(updated)
    return updated


def _find_orphaned_apple_iaps(
    apple_products: List[Dict[str, Any]], 
    google_products: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Find Apple IAPs without Google counterparts and Google-only IAPs.

    Matching rules:
    - Match by product key extracted from SKU/Product ID
    - Apple: ios_dg_XXXX_YYYY -> key: dg_XXXX
    - Google: aos_dg_XXXX_YYYY -> key: dg_XXXX
    - Match if: dg_XXXX parts are the same (e.g., ios_dg_1098_550 matches aos_dg_1098_550)
    - Also handles cases like ios_dg_1098_550 and aos_dg_1098_550f (same key: dg_1098)

    Returns:
        (apple_orphaned, google_only)
    """
    # Build map of Google products by key
    google_map: Dict[str, List[Dict[str, Any]]] = {}
    google_no_key: List[Dict[str, Any]] = []
    google_total_count = 0

    for product in google_products:
        google_total_count += 1
        sku = (product.get("sku") or "").strip()
        if not sku:
            google_no_key.append(product)
            logger.debug("Google Play IAP with no SKU found: %s", product.get("id", "unknown"))
            continue

        key = _extract_product_key(sku)
        if key:
            google_map.setdefault(key, []).append(product)
            logger.debug("Google Play IAP SKU: %s -> key: %s", sku, key)
        else:
            google_no_key.append(product)
            logger.debug("Google Play IAP SKU doesn't match expected pattern: %s (parts: %s)", sku, sku.split('_') if sku else [])

    # Find Apple IAPs without matching Google IAPs
    orphaned: List[Dict[str, Any]] = []
    matched_keys: set[str] = set()

    for product in apple_products:
        apple_sku = (product.get("productId") or product.get("sku") or "").strip()
        if not apple_sku:
            orphaned.append(product)
            logger.debug("Apple IAP with no SKU/Product ID found: %s", product.get("id", "unknown"))
            continue

        key = _extract_product_key(apple_sku)
        if not key:
            orphaned.append(product)
            logger.debug("Apple IAP SKU doesn't match expected pattern: %s (parts: %s)", apple_sku, apple_sku.split('_') if apple_sku else [])
            continue

        if key not in google_map:
            orphaned.append(product)
            logger.debug("Apple IAP has no matching Google Play IAP: %s (key: %s, available keys: %s)", 
                        apple_sku, key, list(google_map.keys())[:10])  # 처음 10개만 로깅
        else:
            matched_keys.add(key)
            logger.debug("Apple IAP has matching Google Play IAP: %s (key: %s)", apple_sku, key)

    # Build Google-only list (canonicalized products)
    google_only: List[Dict[str, Any]] = []

    # Products that couldn't be keyed
    for product in google_no_key:
        canonical = _canonicalize_google_product(product)
        if canonical:
            google_only.append(canonical)

    # Products whose key wasn't matched by any Apple IAP
    unmatched_google_keys = set(google_map.keys()) - matched_keys
    for key in unmatched_google_keys:
        products_by_key = google_map[key]
        for product in products_by_key:
            canonical = _canonicalize_google_product(product)
            if canonical:
                google_only.append(canonical)

    # Log detailed statistics
    logger.info(
        "IAP matching results: %d apple-orphaned, %d google-only, %d matched keys (out of %d Apple IAPs, %d Google IAPs)",
        len(orphaned),
        len(google_only),
        len(matched_keys),
        len(apple_products),
        google_total_count
    )
    logger.debug(
        "Matching details: Google keys=%d, Matched keys=%d, Unmatched Google keys=%d, Google no-key=%d",
        len(google_map),
        len(matched_keys),
        len(unmatched_google_keys),
        len(google_no_key)
    )

    return orphaned, google_only


@app.get("/api/apple/inapp/orphaned")
async def api_apple_orphaned_iaps():
    """Get list of Apple IAPs that don't have matching Google Play IAPs.
    
    Two IAPs are considered matching if their product keys match:
    - Apple: ios_dg_XXXX_YYYY -> key: dg_XXXX
    - Google: aos_dg_XXXX_YYYY -> key: dg_XXXX
    - Match if: dg_XXXX parts are the same (e.g., ios_dg_1098_550 matches aos_dg_1098_550)
    - Also handles cases like ios_dg_1098_550 and aos_dg_1098_550f (same key: dg_1098)
    
    Only returns IAPs that are NOT present in Google Play (orphaned IAPs).
    """
    try:
        # Get all Apple IAPs (from cache, no refresh needed)
        apple_products = await _run_in_thread(
            get_cached_products, APPLE_STORE, _fetch_apple_products, False
        )
        
        # Get all Google IAPs - refresh to ensure we have the latest data
        # This ensures we compare against the current Google Play IAP list
        google_products = await _run_in_thread(
            refresh_products_from_remote, GOOGLE_STORE, _fetch_google_products
        )
        
        # Find orphaned Apple IAPs and Google-only IAPs
        orphaned, google_only = _find_orphaned_apple_iaps(apple_products, google_products)
        
        protected_ids = _get_protected_apple_ids()
        protected_items: List[Dict[str, Any]] = []
        if protected_ids:
            product_map = {
                (item.get("productId") or item.get("sku") or ""): item
                for item in apple_products
                if item
            }
            for pid in sorted(protected_ids):
                item = product_map.get(pid)
                if item:
                    protected_items.append(item)

        filtered_orphaned = [
            item
            for item in orphaned
            if (item.get("productId") or item.get("sku") or "") not in protected_ids
        ]

        return {
            "items": filtered_orphaned,
            "totalCount": len(filtered_orphaned),
            "googleOnly": google_only,
            "protectedItems": protected_items,
            "protectedIds": sorted(protected_ids),
        }
    except Exception as exc:
        logger.exception("Failed to find orphaned Apple IAPs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/cleanup")
async def api_apple_cleanup(request: Request):
    """Delete or remove from sale selected Apple IAPs that don't have matching Google Play IAPs.
    
    Request body should contain:
    {
        "productIds": ["ios_dg_1126_99", "ios_dg_1103_1100", ...],
        "action": "delete" | "remove_from_sale"
    }
    
    action:
    - "delete": Permanently delete the IAP (cannot be undone, productId cannot be reused)
    - "remove_from_sale": Set clearedForSale=False (prevents new purchases, maintains access for existing purchasers)
    """
    try:
        payload = await request.json()
        product_ids = payload.get("productIds", [])
        action = payload.get("action", "delete")  # Default to delete for backward compatibility
        
        if not product_ids:
            raise HTTPException(status_code=400, detail="처리할 상품 ID가 없습니다.")
        
        if action not in ("delete", "remove_from_sale"):
            raise HTTPException(
                status_code=400,
                detail=f"잘못된 action입니다. 'delete' 또는 'remove_from_sale'만 허용됩니다."
            )
        
        # Verify these are actually orphaned (safety check)
        # Refresh Google Play IAP list to ensure we have the latest data
        apple_products = await _run_in_thread(
            get_cached_products, APPLE_STORE, _fetch_apple_products, False
        )
        google_products = await _run_in_thread(
            refresh_products_from_remote, GOOGLE_STORE, _fetch_google_products
        )
        
        orphaned, _ = _find_orphaned_apple_iaps(apple_products, google_products)
        protected_ids = _get_protected_apple_ids()
        orphaned_map = {
            (item.get("productId") or item.get("sku") or "").strip(): item
            for item in orphaned
            if item
            and (item.get("productId") or item.get("sku") or "").strip() not in protected_ids
        }
        
        # Verify all requested IDs are actually orphaned
        invalid_ids = [pid for pid in product_ids if pid not in orphaned_map]
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"다음 상품 ID들은 Google Play에 매칭되는 상품이 있습니다: {', '.join(invalid_ids[:5])}"
            )
        
        # Process IAPs
        results = {"processed": 0, "failed": 0}
        failed_items: List[Dict[str, Any]] = []

        def _process_iap(product_id: str) -> Tuple[bool, str, Optional[str]]:
            """Process a single IAP. Returns (success, product_id, error)."""
            try:
                # Find the IAP to get the numeric ID (inapp_id)
                item = orphaned_map.get(product_id)
                if not item:
                    return (False, product_id, "IAP를 찾을 수 없습니다.")
                
                # Get the numeric ID (inapp_id) - needed for API calls
                inapp_id = item.get("id") or ""
                if not inapp_id:
                    # Try to find it by product_id
                    try:
                        detail = _find_apple_inapp(product_id)
                        inapp_id = detail.get("id") or ""
                    except Exception:
                        pass
                
                if not inapp_id:
                    return (False, product_id, "IAP ID를 찾을 수 없습니다.")
                
                if action == "delete":
                    # Delete the IAP
                    delete_apple_inapp_purchase(inapp_id)
                    delete_cached_product(APPLE_STORE, product_id)
                    logger.info("Deleted orphaned Apple IAP: %s (id: %s)", product_id, inapp_id)
                elif action == "remove_from_sale":
                    # Remove from sale
                    updated = remove_apple_inapp_purchase_from_sale(inapp_id)
                    # Update cache with updated IAP info
                    summary = _to_apple_summary(updated)
                    upsert_cached_product(APPLE_STORE, summary)
                    logger.info("Removed orphaned Apple IAP from sale: %s (id: %s)", product_id, inapp_id)
                
                return (True, product_id, None)
            except Exception as exc:
                return (False, product_id, str(exc))

        def _process_all() -> Tuple[Dict[str, int], List[Dict[str, Any]], List[str]]:
            import concurrent.futures

            if not product_ids:
                return results, failed_items, []

            max_workers = min(5, len(product_ids))  # Limit concurrency to avoid rate limits
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_id = {
                    executor.submit(_process_iap, product_id): product_id
                    for product_id in product_ids
                }

                successful_ids: List[str] = []
                for future in concurrent.futures.as_completed(future_to_id):
                    pid = future_to_id[future]
                    try:
                        success, product_id, error = future.result()
                    except Exception as exc:
                        success, product_id, error = False, pid, str(exc)

                    if success:
                        results["processed"] += 1
                        successful_ids.append(product_id)
                    else:
                        results["failed"] += 1
                        failed_items.append({
                            "productId": product_id,
                            "error": error or "Unknown error"
                        })
                        logger.error("Failed to %s Apple IAP %s: %s", action, product_id, error)

            return results, failed_items, successful_ids

        results, failed_items, successful_ids = await _run_in_thread(_process_all)

        if action == "delete" and successful_ids:
            _remove_protected_apple_ids(successful_ids)

        response_data = {
            "status": "ok" if not failed_items else "partial",
            "summary": results,
            "action": action
        }
        
        if failed_items:
            response_data["failed_items"] = failed_items
        
        return response_data
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to cleanup orphaned Apple IAPs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/apple/inapp/cleanup/protect")
async def api_apple_cleanup_protect(payload: AppleProtectionRequest):
    try:
        product_ids = {pid.strip() for pid in payload.product_ids if pid and pid.strip()}
        if not product_ids:
            raise HTTPException(status_code=400, detail="보호할 상품 ID를 입력해주세요.")

        updated = _add_protected_apple_ids(product_ids)
        logger.info("Protected %d Apple IAPs (total protected: %d)", len(product_ids), len(updated))

        return {
            "status": "ok",
            "protectedIds": sorted(updated),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to protect Apple IAPs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/apple/inapp/cleanup/unprotect")
async def api_apple_cleanup_unprotect(payload: AppleProtectionRequest):
    try:
        product_ids = {pid.strip() for pid in payload.product_ids if pid and pid.strip()}
        if not product_ids:
            raise HTTPException(status_code=400, detail="보호 해제할 상품 ID를 입력해주세요.")

        updated = _remove_protected_apple_ids(product_ids)
        logger.info("Unprotected %d Apple IAPs (total protected: %d)", len(product_ids), len(updated))

        return {
            "status": "ok",
            "protectedIds": sorted(updated),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to unprotect Apple IAPs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
